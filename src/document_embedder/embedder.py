import os
import time

import pandas as pd
from natsort import natsorted
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.document_embedder.config import Config
from src.utils.embeddings_store import EmbeddingsStore
from src.utils.logger_setup import get_logger, configure_file_logger
from src.utils.utils import elapsed_time_to_string

logger = get_logger(__name__)

# the number of documents processed per batch
BATCH_SIZE = 10_000


def run(config: Config):
    # setup logfile
    logfile = os.path.join(config.work_dir, f"embedder_{config.embeddings_dir}.log")
    configure_file_logger(logger, logfile)
    # generate embeddings for documents
    _embed_documents(documents_dir=config.documents, work_dir=config.work_dir, embeddings_dir=config.embeddings_dir)


def _extract_id_from_filename(filename: str) -> int:
    """
    Extract textfile id from filename.
    :param filename: the filename (not including the path)
    :return: the integer present in the filename.
    """
    id_str = filename.split('_')[1]
    id_str = id_str.split('.')[0]
    return int(id_str)


def _embed_documents(documents_dir: str, work_dir: str, embeddings_dir: str):
    # Load a pretrained Sentence Transformer model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    logger.info(
        f"Loaded model '{model_name}' (max sequence length = {model.max_seq_length}, embedding dimension = {model.get_sentence_embedding_dimension()})")

    # List documents
    text_files = [f for f in os.listdir(documents_dir) if
                  f.endswith('.txt') and os.path.isfile(os.path.join(documents_dir, f))]
    text_files = natsorted(text_files)

    logger.info(f"Found {len(text_files)} documents.")

    # Create embeddings store instance to save embeddings
    embeddings_store = os.path.join(work_dir, embeddings_dir)
    embeddings_store = EmbeddingsStore(embeddings_store)
    # Clear out embeddings store
    embeddings_store.cleanup()

    # start timer
    start_time = time.time()

    # Read and encode documents in batch
    nr_batches = -(len(text_files) // -BATCH_SIZE)
    for i in tqdm(range(0, len(text_files), BATCH_SIZE), desc="Generating embeddings", unit=f"{BATCH_SIZE} documents",
                  total=nr_batches):
        # read documents and collect document text and document ids
        documents = []
        document_ids = []
        for text_file in text_files[i:i + BATCH_SIZE]:
            document_ids.append(_extract_id_from_filename(text_file))
            full_path = os.path.join(documents_dir, text_file)
            with open(full_path, "r", encoding='utf-8') as f:
                documents.append(f.read())
        # generate embeddings for documents
        embeddings = model.encode(documents, batch_size=32, show_progress_bar=False, convert_to_numpy=True,
                                  precision="float32", normalize_embeddings=True)
        # store document ids and document embeddings in vect
        df = pd.DataFrame(embeddings)
        df.columns = df.columns.map(str)
        df["id"] = document_ids
        embeddings_store.write(df)

    # log elapsed time to generate embeddings
    logger.info(f"Generated embeddings for all documents in: {elapsed_time_to_string(time.time() - start_time)}")

    logger.info(f"Created vector store at '{embeddings_store.path}'.")
    # remove old backup versions to reduce size on disk (lance feature)
    embeddings_store.cleanup()
