"""
Offline evaluation metrics for the recommendation engine.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from newsrec import config
from newsrec.recommender import ContentRecommender

# Configure logging
logger = logging.getLogger(__name__)

def train_test_split_chrono(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame chronologically.

    Args:
        df: The DataFrame to split.
        test_size: The proportion of the dataset to include in the test split.

    Returns:
        A tuple containing the train and test DataFrames.
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def precision_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Calculates precision at k.
    """
    return len(set(recommended_items[:k]) & set(relevant_items)) / k

def recall_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Calculates recall at k.
    """
    return len(set(recommended_items[:k]) & set(relevant_items)) / len(relevant_items)

def mean_average_precision(recommended_items: List[int], relevant_items: List[int]) -> float:
    """
    Calculates Mean Average Precision (MAP).
    """
    ap = 0.0
    hits = 0
    for i, p in enumerate(recommended_items):
        if p in relevant_items:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(relevant_items) if relevant_items else 0.0

def mean_reciprocal_rank(recommended_items: List[int], relevant_items: List[int]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR).
    """
    for i, p in enumerate(recommended_items):
        if p in relevant_items:
            return 1 / (i + 1)
    return 0.0

def evaluate(
    recommender: ContentRecommender,
    df_holdout: pd.DataFrame,
    history_len: int = config.EVALUATION_CONFIG["history_length_for_eval"],
) -> Dict[str, float]:
    """
    Evaluates the recommender on a hold-out set.

    Args:
        recommender: The fitted recommender to evaluate.
        df_holdout: The hold-out DataFrame.
        history_len: The number of items to use as user history for evaluation.

    Returns:
        A dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating recommender with history length {history_len}...")
    metrics = {f"precision@{k}": [] for k in config.EVALUATION_CONFIG["metrics_k_values"]}
    metrics.update({f"recall@{k}": [] for k in config.EVALUATION_CONFIG["metrics_k_values"]})
    metrics["map"] = []
    metrics["mrr"] = []

    for _, group in df_holdout.groupby(np.arange(len(df_holdout)) // (history_len + 1)):
        if len(group) <= history_len:
            continue

        history_df = group.iloc[:history_len]
        relevant_items = group.iloc[history_len:]["id"].tolist()
        user_history = history_df["id"].tolist()

        recommendations = recommender.recommend(user_history, k=max(config.EVALUATION_CONFIG["metrics_k_values"]))[0]

        for k in config.EVALUATION_CONFIG["metrics_k_values"]:
            metrics[f"precision@{k}"].append(precision_at_k(recommendations, relevant_items, k))
            metrics[f"recall@{k}"].append(recall_at_k(recommendations, relevant_items, k))

        metrics["map"].append(mean_average_precision(recommendations, relevant_items))
        metrics["mrr"].append(mean_reciprocal_rank(recommendations, relevant_items))

    return {k: np.mean(v) for k, v in metrics.items()}
