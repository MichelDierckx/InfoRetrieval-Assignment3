import os
from typing import List

import pandas
import pandas as pd

from src.document_embedder.config import Config
from src.utils.logger_setup import get_logger

logger = get_logger(__name__)


def run(config: Config):
    # compute mean recall at k and mean precision at k
    evaluation_results = _compute_recall_and_precision(ranking=config.ranking, gt=config.gt, k_values=config.k)
    # write computed evaluation results to disk
    _write_results(evaluation_results=evaluation_results, work_dir=config.work_dir, eval_filename=config.eval_filename)


def _compute_recall_and_precision(ranking: str, gt: str, k_values: List[int]):
    # Load ground truth and rankings
    ground_truth = pd.read_csv(gt)
    rankings = pd.read_csv(ranking)
    k_values.sort()

    # NOTE: order of documents within a group is preserved https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
    # convert to dict (query_nr, set(doc_nr)), set is used since order DOES NOT matter
    gt_dict = ground_truth.groupby("Query_number")["doc_number"].apply(set).to_dict()
    # convert to dict (query_nr, list(doc_nr)), set is used since order DOES matter
    rank_dict = rankings.groupby("Query_number")["doc_number"].apply(list).to_dict()

    # check whether both dicts contain the same set of queries
    assert (sorted(gt_dict.keys()) == sorted(gt_dict.keys()))

    # calculate precision at k and recall at k for every query for each k
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}

    # calculate precision at k and recall at k for every query
    for query, retrieved_docs in rank_dict.items():
        # Get ground truth for the current query
        relevant_docs = gt_dict.get(query, set())
        # Calculate precision and recall for each k
        for k in k_values:
            # Limit retrieved docs to top k
            top_k_docs = retrieved_docs[:k]
            # Calculate intersection with relevant docs
            retrieved_relevant_docs = set(top_k_docs) & relevant_docs

            # Calculate precision@k
            precision = len(retrieved_relevant_docs) / k
            precision_at_k[k].append(precision)

            # Calculate recall@k
            recall = len(retrieved_relevant_docs) / len(relevant_docs) if relevant_docs else 0
            recall_at_k[k].append(recall)

    # Compute mean metrics for each k
    mean_precision = {k: sum(precision_at_k[k]) / len(precision_at_k[k]) for k in k_values}
    mean_recall = {k: sum(recall_at_k[k]) / len(recall_at_k[k]) for k in k_values}

    # get basename of rankings file
    ranking_basename = os.path.splitext(os.path.basename(ranking))[0]

    # create a pandas df to return results
    results = []
    for k in k_values:
        results.append({
            "ranking_filename": ranking_basename,
            "k": k,
            "mean precision": mean_precision[k],
            "mean recall": mean_recall[k]
        })
    results_df = pd.DataFrame(results)
    return results_df


def _write_results(evaluation_results: pandas.DataFrame, work_dir: str, eval_filename: str):
    file_path = os.path.join(work_dir, eval_filename + ".csv")
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    if file_exists:
        # Read existing data
        existing_data = pd.read_csv(file_path)

        # Concatenate the existing data with the new results
        combined_data = pd.concat([existing_data, evaluation_results], ignore_index=True)

        # overwrite results with the same k and ranking_filename
        combined_data = combined_data.drop_duplicates(subset=['ranking_filename', 'k'], keep='last')
    else:
        # If file does not exist, use the results_df as is
        combined_data = evaluation_results

    # Write the DataFrame to CSV, creating it if it doesn't exist
    combined_data.to_csv(file_path, index=False)
