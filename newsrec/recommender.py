"""
Core content-based recommendation engine.
"""
import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from newsrec import config
from newsrec.similarity import SimilarityScorer
from newsrec.text.preprocess import preprocess_text
from newsrec.text.vectorizers.base import VectorizerBase

# Configure logging
logger = logging.getLogger(__name__)

class ContentRecommender:
    """
    A content-based recommendation engine.
    """

    def __init__(self, vectorizer: VectorizerBase, scorer: str = "cosine"):
        """
        Initializes the ContentRecommender.

        Args:
            vectorizer: An instance of a fitted vectorizer.
            scorer: The similarity scoring strategy to use.
        """
        self.vectorizer = vectorizer
        self.scorer = SimilarityScorer(strategy=scorer)
        self.item_vectors = None
        self.item_ids = None

    def fit(self, df: pd.DataFrame, text_field: str = "content") -> "ContentRecommender":
        """
        Fits the recommender to the dataset.

        Args:
            df: The DataFrame containing the item data.
            text_field: The name of the column containing the text to be vectorized.

        Returns:
            The fitted recommender instance.
        """
        logger.info(f"Fitting recommender with text from '{text_field}'...")
        processed_text = df[text_field].apply(preprocess_text)
        self.item_vectors = self.vectorizer.fit(processed_text.tolist()).transform(processed_text.tolist())
        self.item_ids = df["id"].tolist()
        logger.info("Recommender fitting complete.")
        return self

    def _create_user_profile(self, user_history_vectors: np.ndarray) -> np.ndarray:
        """
        Creates a user profile vector from their history.

        Args:
            user_history_vectors: A matrix of item vectors from the user's history.

        Returns:
            A single vector representing the user's profile.
        """
        strategy = config.RECOMMENDER_CONFIG.get("user_profile_strategy", "mean")
        if strategy == "mean":
            return np.mean(user_history_vectors, axis=0)
        elif strategy == "weighted_average":
            alpha = config.RECOMMENDER_CONFIG.get("user_profile_weighted_alpha", 0.1)
            weights = np.exp(-alpha * np.arange(len(user_history_vectors)))
            return np.average(user_history_vectors, axis=0, weights=weights)
        else:
            raise ValueError(f"Unsupported user profile strategy: {strategy}")

    def recommend(
        self, user_history: Union[List[str], List[int]], k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Recommends items for a user based on their history.

        Args:
            user_history: A list of item IDs from the user's history.
            k: The number of recommendations to return.

        Returns:
            A ranked list of (item_id, similarity_score) tuples.
        """
        history_indices = [self.item_ids.index(item_id) for item_id in user_history if item_id in self.item_ids]
        if not history_indices:
            logger.warning("User history contains no known items. Cannot make recommendations.")
            return []

        user_history_vectors = self.item_vectors[history_indices]
        user_profile_vector = self._create_user_profile(user_history_vectors)

        scores = self.scorer.score(user_profile_vector, self.item_vectors)

        # Exclude items already in the user's history
        scores[history_indices] = -np.inf

        # Get top k recommendations
        top_k_indices = np.argsort(scores)[::-1][:k]
        recommendations = [(self.item_ids[i], scores[i]) for i in top_k_indices]

        return recommendations
