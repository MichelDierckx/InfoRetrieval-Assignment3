import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.document_embedder.config import Config
from src.utils.embeddings_store import EmbeddingsStore
from src.utils.logger_setup import get_logger

logger = get_logger(__name__)


def run(config: Config):
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
    logger.debug(
        f"Loaded model '{model_name}' (max sequence length = {model.max_seq_length}, embedding dimension = {model.get_sentence_embedding_dimension()})")

    # List documents
    text_files = [f for f in os.listdir(documents_dir) if
                  f.endswith('.txt') and os.path.isfile(os.path.join(documents_dir, f))]
    logger.info(f"Found {len(text_files)} documents.")

    vector_store_path = os.path.join(work_dir, embeddings_dir)
    # Clear lance dataset if exists
    vector_store = EmbeddingsStore(vector_store_path)
    vector_store.cleanup()

    # Encode documents in batch
    batch_size = 10_000
    for i in tqdm(range(0, len(text_files), batch_size), desc="Generating embeddings", unit=f"{batch_size} documents"):
        documents = []
        document_ids = []
        for text_file in text_files[i:i + batch_size]:
            document_ids.append(_extract_id_from_filename(text_file))
            full_path = os.path.join(documents_dir, text_file)
            with open(full_path, "r", encoding='utf-8') as f:
                documents.append(f.read())
        embeddings = model.encode(documents, batch_size=32, show_progress_bar=True, convert_to_numpy=True,
                                  precision="float32", normalize_embeddings=True)
        df = pd.DataFrame(embeddings)
        df.columns = df.columns.map(str)
        df["id"] = document_ids
        vector_store.write(df)
    vector_store.cleanup()
