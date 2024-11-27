import os
from datetime import timedelta

import lance
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.ranker.config import Config
from src.utils.logger_setup import get_logger

logger = get_logger(__name__)


def run(config: Config):
    _index_documents(documents_dir=config.documents, work_dir=config.work_dir)


class VectorStore:
    def __init__(self, path):
        self.path = path
        self.cleanup()
        self.created = False

    def cleanup(self):
        try:
            ds = lance.dataset(self.path)
        except ValueError:
            return
        time_delta = timedelta(microseconds=1)
        ds.cleanup_old_versions(older_than=time_delta)

    def write(self, data: pd.DataFrame):
        if not self.created:
            logger.debug(f"Creating lance dataset at '{self.path}'.")
            lance.write_dataset(
                data,
                self.path,
                mode="overwrite",
                data_storage_version="stable",
            )
            self.created = True
        lance.write_dataset(
            data,
            self.path,
            mode="append",
            data_storage_version="stable",
        )


def _index_documents(documents_dir: str, work_dir: str):
    # Load a pretrained Sentence Transformer model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    logger.debug(f"Loaded model '{model_name}' (max sequence length = {model.max_seq_length})")

    # List documents
    text_files = [f for f in os.listdir(documents_dir) if
                  f.endswith('.txt') and os.path.isfile(os.path.join(documents_dir, f))]
    logger.info(f"Found {len(text_files)} documents.")

    vector_store_path = os.path.join(work_dir, "document_embeddings")
    # Clear lance dataset if exists
    vector_store = VectorStore(vector_store_path)

    # Encode documents in batch
    batch_size = 10_000
    for i in tqdm(range(0, len(text_files), batch_size), desc="Generating embeddings", unit=f"{batch_size} documents"):
        documents = []
        for text_file in text_files[i:i + batch_size]:
            full_path = os.path.join(documents_dir, text_file)
            with open(full_path, "r", encoding='utf-8') as f:
                documents.append(f.read())
        embeddings = model.encode(documents, batch_size=32, show_progress_bar=True, convert_to_numpy=True,
                                  precision="float32", normalize_embeddings=True)
        df = pd.DataFrame(embeddings)
        vector_store.write(df)
    vector_store.cleanup()
