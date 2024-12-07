import math
import os
import time
from typing import List

import faiss
import kneed
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.clustering_tuner.config import Config
from src.utils.embeddings_store import EmbeddingsStore
from src.utils.logger_setup import get_logger, configure_file_logger
from src.utils.utils import elapsed_time_to_string

logger = get_logger(__name__)


def run(config: Config):
    # setup logfile
    embeddings_name = os.path.basename(os.path.normpath(config.embeddings))
    logfile = os.path.join(config.work_dir, f"clustering_tuner_{embeddings_name}.log")
    configure_file_logger(logger, logfile)
    # create an index for the provided embeddings
    _find_optimal_nr_clusters(embeddings_dir=config.embeddings, work_dir=config.work_dir, step_size=config.step_size)


def _find_optimal_nr_clusters(embeddings_dir: str, work_dir: str, step_size: int):
    # create embeddings store instance to access document embeddings
    embeddings_store = EmbeddingsStore(embeddings_dir)
    embeddings_store.created = True
    _elbow_method(embeddings_store=embeddings_store, embeddings_dir=embeddings_dir, work_dir=work_dir,
                  step_size=step_size)


def _elbow_method(embeddings_store: EmbeddingsStore, embeddings_dir: str, work_dir: str, step_size: int):
    # retrieve embeddings size
    embeddings_size = embeddings_store.get_embeddings_size()

    # range of clusters to explore (https://arxiv.org/pdf/2401.08281, https://github.com/facebookresearch/faiss/issues/112, https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids)
    nr_embeddings = embeddings_store.nr_embeddings()
    # faiss recommends 4*sqrt(n) number of clusters for less than 1M vectors, so we explore from 1*sqrt(n) to 7*sqrt(n)
    # nlist_min = math.ceil(1 * math.sqrt(nr_embeddings)
    nlist_min = 100
    # at least 39*nr_clusters training points required, we take 50 as minimum to be safe
    nlist_max = min(math.ceil(18 * math.sqrt(nr_embeddings)), nr_embeddings // 39)

    # distance metric (equal to cosine similarity if embeddings are normalized)
    metric = faiss.METRIC_INNER_PRODUCT

    nlists = []
    sample_sizes = []
    average_distances_results = []
    average_search_time_results = []

    start_testing_time = time.time()
    logger.info(
        f"Initiating elbow method analysis: testing cluster counts from {nlist_min} to {nlist_max} with a step size of {step_size}.")

    # sample_size should at least be 39*nr_clusters
    sample_size = min(39 * nlist_max, nr_embeddings)
    # retrieve samples from embeddings storage
    samples = embeddings_store.sample(sample_size)

    # try different number of clusters (in steps of 100)
    for nlist in tqdm(range(nlist_min, nlist_max + 1, step_size), desc="Testing different numbers of clusters",
                      unit="test"):
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(embeddings_size), embeddings_size, nlist, metric
        )
        # determine centroids
        index.train(samples)
        # retrieve centroids and add them as data points in the index
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        index.add(centroids)

        start_time = time.perf_counter()
        # retrieve distances for training samples to every centroid
        D, I = index.search(samples, 1)
        search_time_micros = (time.perf_counter() - start_time) * 1_000_000

        # D contains cosine similarity, convert to cosine distance
        D = 1.0 - D

        # https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
        average_distances_to_centroid = np.mean(D)
        # average search time in ms
        average_search_time = (search_time_micros / sample_size)

        # save results
        nlists.append(nlist)
        sample_sizes.append(sample_size)
        average_distances_results.append(average_distances_to_centroid)
        average_search_time_results.append(average_search_time)

    # use kneedle algorithm to find elbow point
    elbow_point = _find_elbow_point(nlists=nlists, average_distances=average_distances_results)
    logger.info(f"Optimal number of clusters according to elbow method = {elbow_point}.")

    logger.info(f"Finished elbow analysis in {elapsed_time_to_string(time.time() - start_testing_time)}")
    # write results to disk
    embeddings_name = os.path.basename(os.path.normpath(embeddings_dir))
    _write_results(nlists=nlists, sample_sizes=sample_sizes, average_distances=average_distances_results,
                   average_search_time_results=average_search_time_results, embeddings_name=embeddings_name,
                   work_dir=work_dir)
    # plot results
    _plot_results(elbow_point=elbow_point, nlists=nlists, average_distances=average_distances_results,
                  average_search_time_results=average_search_time_results,
                  embeddings_name=embeddings_name, work_dir=work_dir)


def _write_results(nlists: List, sample_sizes: List, average_distances: List, average_search_time_results: List,
                   embeddings_name: str,
                   work_dir: str):
    output_file = os.path.join(work_dir, f"{embeddings_name}_elbow_method_results.csv")
    # Create a DataFrame
    df = pd.DataFrame({
        "nlist": nlists,
        "sample_size": sample_sizes,
        "average_centroid_distances": average_distances,
        "average_search_time": average_search_time_results
    })
    df.to_csv(output_file, index=False, mode="w")
    logger.info(f"Saved elbow method statistics to '{output_file}'.")


def _plot_results(elbow_point: int, nlists: List, average_distances: List, average_search_time_results: List,
                  embeddings_name: str,
                  work_dir: str):
    # plot distortion
    output_file_distortion = os.path.join(work_dir, f"{embeddings_name}_distortion.png")
    plt.plot(nlists, average_distances, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.grid()
    plt.vlines(elbow_point, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.savefig(output_file_distortion)
    logger.info(f"Saved distortion graph  to '{output_file_distortion}'.")
    plt.close()

    # plot average search time
    output_file_average_search_time = os.path.join(work_dir, f"{embeddings_name}_search_time.png")
    plt.plot(nlists, average_search_time_results, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average search time (microsecond) for a cluster per query')
    plt.title('Search Time vs Number of Clusters')
    plt.grid()
    plt.savefig(output_file_average_search_time)
    logger.info(f"Saved search time graph  to '{output_file_average_search_time}'.")
    plt.close()


def _find_elbow_point(nlists: List, average_distances: List):
    """
    Determine the elbow point using kneedle algorithm (https://www1.icsi.berkeley.edu/~barath/papers/kneedle-simplex11.pdf)
    :param nlists: number of cluster values (x)
    :param average_distances: average distances to nearest centroids (y)
    :return: elbow point (optimal number of clusters according to elbow method)
    """
    kn = kneed.KneeLocator(
        nlists,
        average_distances,
        curve='convex',
        direction='decreasing',
        interp_method='interp1d',
    )
    knee_point_x = kn.knee
    return knee_point_x
