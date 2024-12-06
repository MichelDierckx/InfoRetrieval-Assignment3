import math
import os
import time

import faiss
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.document_embedder.config import Config
from src.utils.embeddings_store import EmbeddingsStore
from src.utils.logger_setup import get_logger, configure_file_logger

logger = get_logger(__name__)

# the number of embeddings added to the index per batch
BATCH_SIZE = 10_000


def run(config: Config):
    # setup logfile
    logfile = os.path.join(config.work_dir, f"clustering_tuner_{config.index_filename}.log")
    configure_file_logger(logger, logfile)
    # create an index for the provided embeddings
    _find_optimal_nr_clusters(embeddings_dir=config.embeddings, work_dir=config.work_dir, step_size=config.step_size)


def _find_optimal_nr_clusters(embeddings_dir: str, work_dir: str, step_size: int):
    # create embeddings store instance to access document embeddings
    embeddings_store = EmbeddingsStore(embeddings_dir)
    embeddings_store.created = True
    _elbow_method(embeddings_store=embeddings_store, embeddings_dir=embeddings_dir, work_dir=work_dir,
                  step_size=step_size)


def _elbow_method(embeddings_store: EmbeddingsStore, embeddings_dir: str, work_dir: str, step_size: int) -> faiss.Index:
    # retrieve embeddings size
    embeddings_size = embeddings_store.get_embeddings_size()

    # range of clusters to explore (https://arxiv.org/pdf/2401.08281, https://github.com/facebookresearch/faiss/issues/112, https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids)
    nr_embeddings = embeddings_store.nr_embeddings()
    # faiss recommends 4*sqrt(n) number of clusters for less than 1M vectors, so we explore from 1*sqrt(n) to 7*sqrt(n)
    # nlist_min = math.ceil(1 * math.sqrt(embeddings_size))
    nlist_min = 1
    # at least 39*nr_clusters training points required, we take 50 as minimum to be safe
    nlist_max = min(math.ceil(7 * math.sqrt(embeddings_size)), len(nr_embeddings) // 50)

    # distance metric (equal to cosine similarity if embeddings are normalized)
    metric = faiss.METRIC_INNER_PRODUCT

    nlists = []
    sample_sizes = []
    wcss_results = []
    average_search_time_results = []

    # try different number of clusters (in steps of 100)
    for nlist in range(nlist_min, nlist_max + 1, step_size):
        # minimum number of samples is 39
        sample_size = min(50 * nlist, nr_embeddings)
        # retrieve samples from embeddings storage
        samples = embeddings_store.sample(sample_size)
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(embeddings_size), embeddings_size, nlist, metric
        )
        # determine centroids
        index.train(samples)
        # retrieve centroids and add them as data points in the index
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        index.add(centroids)

        start_time = time.time()
        # retrieve distances for training samples to every centroid
        D, I = index.search(samples, 1)
        search_time = time.time() - start_time

        # https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
        wcss = np.sum(D ** 2)
        # average search time in ms
        average_search_time = int((search_time / sample_size) * 1000)

        # save results
        nlists.append(nlist)
        sample_sizes.append(sample_size)
        wcss_results.append(wcss)
        average_search_time_results.append(average_search_time)

    # write results to disk
    _write_results(nlists=nlists, sample_sizes=sample_sizes, wcss_results=wcss_results,
                   average_search_time_results=average_search_time_results, embeddings_dir=embeddings_dir,
                   work_dir=work_dir)
    # plot results
    _plot_results(nlists=nlists, wcss_results=wcss_results, average_search_time_results=average_search_time_results,
                  embeddings_dir=embeddings_dir, work_dir=work_dir)


def _write_results(nlists: [], sample_sizes: [], wcss_results: [], average_search_time_results: [], embeddings_dir: str,
                   work_dir: str):
    output_file = os.path.join(work_dir, f"{embeddings_dir}_elbow_method_results.csv")
    # Create a DataFrame
    df = pd.DataFrame({
        "nlist": nlists,
        "sample_size": sample_sizes,
        "wcss": wcss_results,
        "average_search_time": average_search_time_results
    })
    df.to_csv(output_file, index=False)


def _plot_results(nlists: [], wcss_results: [], average_search_time_results: [], embeddings_dir: str,
                  work_dir: str):
    # plot inertia
    output_file_inertia = os.path.join(work_dir, f"{embeddings_dir}_inertia.png")
    plt.plot(nlists, wcss_results, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.grid()
    plt.savefig(output_file_inertia)
    plt.close()

    # plot average search time
    output_file_average_search_time = os.path.join(work_dir, f"{embeddings_dir}_search_time.png")
    plt.plot(nlists, average_search_time_results, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average search time (ms) per query')
    plt.title('Search Time vs Number of Clusters')
    plt.grid()
    plt.savefig(output_file_average_search_time)
    plt.close()
