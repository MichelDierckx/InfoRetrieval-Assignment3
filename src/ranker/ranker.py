import os
import time
from typing import Tuple

import faiss
import numpy as np
import numpy.typing as npt
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.evaluator.config import Config
from src.utils.logger_setup import get_logger

logger = get_logger(__name__)

# the number of queries executed on the index per batch
BATCH_SIZE = 1_000


def run(config: Config):
    _rank_documents(query_file=config.queries, index_file=config.index, rankings_filename=config.rankings,
                    work_dir=config.work_dir, k=config.k, n_probes=config.n_probes)


def _rank_documents(query_file: str, index_file: str, rankings_filename: str, work_dir: str, k: int, n_probes: int):
    # Load a pretrained Sentence Transformer model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    logger.info(
        f"Loaded model '{model_name}' (max sequence length = {model.max_seq_length}, embedding dimension = {model.get_sentence_embedding_dimension()})")

    # Load the index from disk
    index: faiss.Index = faiss.read_index(index_file)
    index_type = get_index_type(index)
    logger.info(f"Loaded index of type '{index_type}' from '{index_file}'.")

    # Create empty output file to save computed document rankings for given queries
    headers = ["Query_number", "doc_number"]
    df_headers = pd.DataFrame(columns=headers)
    rankings_filepath = os.path.join(work_dir, rankings_filename + '.csv')
    df_headers.to_csv(rankings_filepath, index=False)

    query_count = 0
    query_time_start = time.time()

    # (wrap generator in a tqdm object for status updates)
    batches = tqdm(pd.read_csv(query_file, chunksize=BATCH_SIZE), desc="Processing queries",
                   unit=f"{BATCH_SIZE} queries", bar_format='{desc}')

    # Read queries in batch, convert to embeddings and query against the index
    for batch in batches:
        # read queries
        queries = batch["Query"].astype(str).tolist()
        query_count += len(queries)
        # generate embeddings for queries
        embeddings = model.encode(queries, batch_size=32, show_progress_bar=False, convert_to_numpy=True,
                                  precision="float32", normalize_embeddings=True)
        # search the index with the generated embeddings
        _, nn_matrix = _query(index=index, query_embeddings=embeddings, k=k, n_probes=n_probes)

        # combine query results with query numbers
        ids = batch["Query number"].to_numpy()
        ids_repeated = np.repeat(ids, nn_matrix.shape[1])
        nn_flat = nn_matrix.flatten()
        df = pd.DataFrame({"Query_number": ids_repeated, "doc_number": nn_flat})

        # append results to output file
        df.to_csv(rankings_filepath, mode='a', index=False, header=False)

        # update status message
        batches.set_description_str(f"Processed {query_count} queries.")
        batches.update(query_count)

    logger.info(f"Processed {query_count} queries in {time.time() - query_time_start} seconds.")
    logger.info(f"Query results have been written to '{rankings_filepath}'.")


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
    # get the index type
    index_type = get_index_type(index)
    # search using the appropriate index type
    match index_type:
        case "IndexFlatIP":
            # brute force search
            D, I = index.search(query_embeddings, k)
        case "IndexIVFFlat":
            # https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-set-nprobe-on-an-opaque-index
            index = faiss.extract_index_ivf(index)
            # number of clusters to visit
            index.nprobe = n_probes
            # search clusters
            D, I = index.search(query_embeddings, k)
        case _:
            raise ValueError(f"Unsupported index type '{index_type}'")
    return D, I


def get_index_type(index: faiss.Index) -> str:
    # sometimes an index is wrapped inside an IndexIDMap to allow for custom index id's
    if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
        return "IndexFlatIP"
    else:
        # not wrapped inside IndexIDMap
        return type(index).__name__
