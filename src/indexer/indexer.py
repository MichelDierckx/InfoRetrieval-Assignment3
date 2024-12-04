from src.document_embedder.config import Config
from src.utils.embeddings_store import EmbeddingsStore
from src.utils.logger_setup import get_logger

logger = get_logger(__name__)


def run(config: Config):
    _create_index(embeddings_dir=config.embeddings, work_dir=config.work_dir, index_filename=config.index_filename)


def _create_index(embeddings_dir: str, work_dir: str, index_filename: str):
    # create embeddings store instance to access document embeddings
    embeddings_store = EmbeddingsStore(embeddings_dir)
    embeddings_store.created = True

    # count how many embeddings need to be indexed
    nr_embeddings = embeddings_store.nr_embeddings()
    embeddings_size = embeddings_store.get_embeddings_size()
    logger.debug(embeddings_size)

    if nr_embeddings <= 10 ** 4:
        pass
    pass


def _create_bruteforce_index():
    pass
    # index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
