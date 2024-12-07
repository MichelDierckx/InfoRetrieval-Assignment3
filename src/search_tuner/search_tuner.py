import os
import time
from typing import Tuple, List

import faiss
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.search_tuner.config import Config
from src.utils.logger_setup import get_logger

logger = get_logger(__name__)


def run(config: Config):
    _find_optimal_nr_probes(query_file=config.queries, index_file=config.index,
                            work_dir=config.work_dir, gt=config.gt, min_nprobes=config.min_nprobes,
                            max_nprobes=config.max_nprobes, step_size=config.step_size)


def _find_optimal_nr_probes(query_file: str, index_file: str, work_dir: str, gt: str, min_nprobes: int,
                            max_nprobes: int, step_size: int):
    # Load a pretrained Sentence Transformer model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    logger.info(
        f"Loaded model '{model_name}' (max sequence length = {model.max_seq_length}, embedding dimension = {model.get_sentence_embedding_dimension()})")

    # Load the index from disk
    index_name = os.path.basename(os.path.normpath(index_file))
    index: faiss.Index = faiss.read_index(index_file)
    logger.info(f"Loaded index from '{index_file}'.")

    # Determine the separator based on the file extension
    file_extension = os.path.splitext(query_file)[1].lower()
    sep = '\t' if file_extension == '.tsv' else ','
    queries_df = pd.read_csv(query_file, sep=sep, nrows=1000)
    nr_queries = queries_df.shape[0]
    # generate embeddings for queries
    queries = queries_df["Query"].astype(str).tolist()
    embeddings = model.encode(queries, batch_size=32, show_progress_bar=False, convert_to_numpy=True,
                              precision="float32", normalize_embeddings=True)
    ids = queries_df["Query number"].to_numpy()

    # Load ground truth
    ground_truth = pd.read_csv(gt)

    nprobe_values = []
    average_search_times = []
    mean_precision_values = []
    mean_recall_values = []

    logger.info(
        f"Initiating elbow method analysis: testing values for number of clusters probed from {min_nprobes} to {max_nprobes} with a step size of {step_size}.")

    for n_probes in tqdm(range(min_nprobes, max_nprobes + 1, step_size),
                         desc="Testing different values for number of clusters probed.",
                         unit="test"):
        # search the index with the generated embeddings
        start_time = time.perf_counter()
        _, nn_matrix = _query(index=index, query_embeddings=embeddings, k=10, n_probes=n_probes)
        search_time_micros = (time.perf_counter() - start_time) * 1_000_000
        average_search_time = search_time_micros / nr_queries

        ids_repeated = np.repeat(ids, nn_matrix.shape[1])
        nn_flat = nn_matrix.flatten()
        df = pd.DataFrame({"Query_number": ids_repeated, "doc_number": nn_flat})
        mean_precision, mean_recall = _compute_recall_and_precision(df, ground_truth)

        nprobe_values.append(n_probes)
        average_search_times.append(average_search_time)
        mean_precision_values.append(mean_precision)
        mean_recall_values.append(mean_recall)

    _write_results(nprobe_values=nprobe_values, average_search_times=average_search_times,
                   mean_precision_values=mean_precision_values, mean_recall_values=mean_recall_values,
                   index_name=index_name, work_dir=work_dir)
    _plot_results(nprobe_values=nprobe_values, average_search_time_values=average_search_times,
                  mean_precision_values=mean_precision_values, mean_recall_values=mean_recall_values,
                  index_name=index_name, work_dir=work_dir)


def _query(index: faiss.Index, query_embeddings: npt.NDArray[np.float32], k: int, n_probes: int) -> Tuple[
    npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """
    Search the index using the embeddings for the queries.
    :param index: a faiss index instance, either of type IndexIDMap(IndexFlatIP) or of type IndexIVFFlat
    :param query_embeddings: the embeddings for the queries
    :param k: the number of neighbours to retrieve for each query
    :param n_probes: the number of clusters to be probed during search (only affects indexes of type IndexIVFFlat)
    :return: a tuple (D, I), where D will contain the distances and I will contain the indices of the neighbours
    """
    index = faiss.extract_index_ivf(index)
    # number of clusters to visit
    index.nprobe = n_probes
    # search clusters
    D, I = index.search(query_embeddings, k)
    return D, I


def _compute_recall_and_precision(rankings: pd.DataFrame, ground_truth: pd.DataFrame):
    k = 10
    # NOTE: order of documents within a group is preserved https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
    # convert to dict (query_nr, set(doc_nr)), set is used since order DOES NOT matter
    gt_dict = ground_truth.groupby("Query_number")["doc_number"].apply(set).to_dict()
    # convert to dict (query_nr, list(doc_nr)), set is used since order DOES matter
    rank_dict = rankings.groupby("Query_number")["doc_number"].apply(list).to_dict()

    # check whether both dicts contain the same set of queries
    assert (sorted(gt_dict.keys()) == sorted(gt_dict.keys()))

    precisions = []
    recalls = []

    # calculate precision at k and recall at k for every query
    for query, retrieved_docs in rank_dict.items():
        # Get ground truth for the current query
        relevant_docs = gt_dict.get(query, set())
        # Limit retrieved docs to top k
        top_k_docs = retrieved_docs[:k]
        # Calculate intersection with relevant docs
        retrieved_relevant_docs = set(top_k_docs) & relevant_docs

        # Calculate precision@k
        precision = len(retrieved_relevant_docs) / k
        precisions.append(precision)

        # Calculate recall@k
        recall = len(retrieved_relevant_docs) / len(relevant_docs) if relevant_docs else 0
        recalls.append(recall)

    # Compute mean metrics for each k
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)

    return mean_precision, mean_recall


def _write_results(nprobe_values: List, average_search_times: List, mean_precision_values: List,
                   mean_recall_values: List,
                   index_name: str,
                   work_dir: str):
    output_file = os.path.join(work_dir, f"{index_name}_nprobe_tuning_results.csv")
    # Create a DataFrame
    df = pd.DataFrame({
        "nprobes": nprobe_values,
        "average_search_time(microseconds)": average_search_times,
        "mean_precision_at_10": mean_precision_values,
        "mean_recall_at_10": mean_recall_values
    })
    df.to_csv(output_file, index=False, mode="w")
    logger.info(f"Saved nprobes statistics to '{output_file}'.")


def _plot_results(nprobe_values: List, average_search_time_values: List, mean_precision_values: List,
                  mean_recall_values: List,
                  index_name: str,
                  work_dir: str):
    # plot mean_precision_at_10
    output_file_precision = os.path.join(work_dir, f"{index_name}_precision.png")
    plt.plot(nprobe_values, mean_precision_values, 'bx-')
    plt.xlabel('Number of probes (nprobes)')
    plt.ylabel('Mean Precision at 10')
    plt.title('Precision vs. Number of probes (nprobes)')
    plt.grid()
    plt.savefig(output_file_precision)
    logger.info(f"Saved precision graph  to '{output_file_precision}'.")
    plt.close()

    # Plot average search times
    output_file_search_time = os.path.join(work_dir, f"{index_name}_average_search_time.png")
    plt.plot(nprobe_values, average_search_time_values, 'bx-')
    plt.xlabel('Number of Probes (nprobes)')
    plt.ylabel('Average Search Time (microseconds)')
    plt.title('Average Search Time vs. Number of Probes (nprobes)')
    plt.grid()
    plt.savefig(output_file_search_time)
    logger.info(f"Saved average search time graph to '{output_file_search_time}'.")
    plt.close()

    # Plot mean recall
    output_file_recall = os.path.join(work_dir, f"{index_name}_mean_recall.png")
    plt.plot(nprobe_values, mean_recall_values, 'bx-')
    plt.xlabel('Number of Probes (nprobes)')
    plt.ylabel('Mean Recall at 10')
    plt.title('Mean Recall at 10 vs. Number of Probes (nprobes)')
    plt.grid()
    plt.savefig(output_file_recall)
    logger.info(f"Saved mean recall graph to '{output_file_recall}'.")
    plt.close()
