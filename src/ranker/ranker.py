import os
import time
from typing import Tuple

import faiss
import numpy as np
import numpy.typing as npt
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.document_embedder.config import Config
from src.utils.logger_setup import get_logger

logger = get_logger(__name__)

# the number of queries executed on the index per batch
BATCH_SIZE = 1_000


def run(config: Config):
    _rank_documents(query_file=config.queries, index_file=config.index, rankings_filename=config.rankings,
                    work_dir=config.work_dir)


def _rank_documents(query_file: str, index_file: str, rankings_filename: str, work_dir: str):
    # Load a pretrained Sentence Transformer model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    logger.info(
        f"Loaded model '{model_name}' (max sequence length = {model.max_seq_length}, embedding dimension = {model.get_sentence_embedding_dimension()})")

    # Load the index from disk
    index: faiss.Index = faiss.read_index(index_file)
    index_type = type(index).__name__
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
        _, nn_matrix = _query(index, embeddings)

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


def _query(index: faiss.Index, query_embeddings: npt.NDArray[np.float32]) -> Tuple[
    npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """
    Search the index using the embeddings for the queries.
    :param index: a faiss index instance, either of type IndexIDMap or of type IndexIVFFlat
    :param query_embeddings: the embeddings for the queries
    :return: a tuple (D, I), where D will contain the distances and I will contain the indices of the neighbours
    """
    # the number of neighbors to retrieve for each query
    nr_neighbours = 10
    # get the index type
    index_type = type(index).__name__
    # search using the appropriate index type

    match index_type:
        case "IndexIDMap":
            D, I = index.search(query_embeddings, nr_neighbours)
        case "IndexIVFFlat":
            index.nprobe = 10
            D, I = index.search(query_embeddings, nr_neighbours)
        case _:
            raise ValueError(f"Unsupported index type '{index_type}'")
    return D, I
