import math
import os
import time
from typing import Optional

import faiss
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.document_embedder.config import Config
from src.utils.embeddings_store import EmbeddingsStore
from src.utils.logger_setup import get_logger, configure_file_logger
from src.utils.utils import elapsed_time_to_string

logger = get_logger(__name__)

# the number of embeddings added to the index per batch
BATCH_SIZE = 10_000


def run(config: Config):
    # setup logfile
    logfile = os.path.join(config.work_dir, f"indexer_{config.index_filename}.log")
    configure_file_logger(logger, logfile)
    # create an index for the provided embeddings
    _index(embeddings_dir=config.embeddings, work_dir=config.work_dir, index_filename=config.index_filename,
           index_type=config.index_type, nlist=config.nlist)


def _index(embeddings_dir: str, work_dir: str, index_filename: str, index_type: Optional[str], nlist: Optional[int]):
    # create embeddings store instance to access document embeddings
    embeddings_store = EmbeddingsStore(embeddings_dir)
    embeddings_store.created = True

    # count how many embeddings need to be indexed
    nr_embeddings = embeddings_store.nr_embeddings()
    logger.info(f"Found {nr_embeddings} embeddings at '{embeddings_dir}'.")

    # start timer
    create_index_time_start = time.time()

    match index_type:
        case "Exhaustive":
            index = _create_bruteforce_index(embeddings_store=embeddings_store)
        case "IVF":
            index = _create_ivf_index(embeddings_store=embeddings_store, nlist=nlist, work_dir=work_dir,
                                      index_filename=index_filename)
        case _:
            if nr_embeddings <= 10 ** 4:
                index = _create_bruteforce_index(embeddings_store=embeddings_store)
            elif nr_embeddings <= 10 ** 6:
                index = _create_ivf_index(embeddings_store=embeddings_store, nlist=nlist, work_dir=work_dir,
                                          index_filename=index_filename)
            else:
                raise ValueError(f'Cannot create index for {nr_embeddings} embeddings')

    # log elapsed time to create index
    logger.info(f"Created index in: {elapsed_time_to_string(time.time() - create_index_time_start)}")

    # write index to file
    index_filename = index_filename + '.index'
    index_path = os.path.join(work_dir, index_filename)
    logger.info(f"Saving index to '{index_path}'")
    faiss.write_index(index, index_path)


def _create_bruteforce_index(embeddings_store: EmbeddingsStore) -> faiss.Index:
    # retrieve embeddings size
    embeddings_size = embeddings_store.get_embeddings_size()
    # create brute-force index (with ids) and using default distance metric (inner product)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings_size))
    logger.info("Creating an exhaustive search index...")
    # add document embeddings to index in batches
    for ids, embeddings in embeddings_store.get_batches(BATCH_SIZE):
        index.add_with_ids(embeddings, ids)
    return index


def _create_ivf_index(embeddings_store: EmbeddingsStore, nlist: Optional[str], work_dir: str,
                      index_filename: str) -> faiss.Index:
    # retrieve embeddings size
    embeddings_size = embeddings_store.get_embeddings_size()
    # number of clusters (https://arxiv.org/pdf/2401.08281, https://github.com/facebookresearch/faiss/issues/112)
    nr_embeddings = embeddings_store.nr_embeddings()
    nlist = math.ceil(4 * math.sqrt(nr_embeddings)) if nlist is None else nlist
    # distance metric (equal to cosine similarity if embeddings are normalized)
    metric = faiss.METRIC_INNER_PRODUCT
    # create IVF index (with ids)
    index = faiss.IndexIVFFlat(
        faiss.IndexFlatIP(embeddings_size), embeddings_size, nlist, metric
    )
    logger.info(f"Creating an IVF index with {nlist} clusters.")
    # count how many embeddings need to be indexed
    nr_embeddings = embeddings_store.nr_embeddings()
    # calculate an appropriate sample size (https://github.com/facebookresearch/faiss/wiki/FAQ)
    sample_size = min(50 * nlist, nr_embeddings)
    # retrieve samples from embeddings storage
    samples = embeddings_store.sample(sample_size)
    # train IVF index (aka determine cluster centroids)
    logger.info(f"Training IVF index on {sample_size} samples.")
    train_time_start = time.time()
    index.train(samples)
    train_duration = time.time() - train_time_start
    logger.info(f"Finished training IVF index in {train_duration} seconds.")
    # add document embeddings to index in batches
    nr_batches = -(nr_embeddings // -BATCH_SIZE)
    for ids, embeddings in tqdm(embeddings_store.get_batches(BATCH_SIZE), desc="Adding embeddings to index",
                                unit=f"{BATCH_SIZE} embeddings", total=nr_batches):
        index.add_with_ids(embeddings, ids)
    # create a scatter plot of the centroids
    _plot_ivf_cluster_centroids(index=index, work_dir=work_dir, index_filename=index_filename)
    return index


def _plot_ivf_cluster_centroids(index: faiss.Index, work_dir: str, index_filename: str):
    # retrieve centroids
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    # reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    centroids_2d = pca.fit_transform(centroids)
    # create a scatter plot
    scatter_plot_path = os.path.join(work_dir, f"indexer_{index_filename}_centroids.png")
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='blue', s=50, label='Centroids')
    plt.title('Centroids Scatter Plot')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid()
    plt.savefig(scatter_plot_path)
    logger.info(f"Saved centroids scatter plot to {scatter_plot_path}.")
