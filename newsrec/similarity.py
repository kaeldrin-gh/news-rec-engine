"""
Similarity scoring module.
"""
import logging
from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from newsrec import config

# Configure logging
logger = logging.getLogger(__name__)

class SimilarityScorer:
    """
    Calculates similarity between vectors using various strategies.
    """

    def __init__(self, strategy: Literal["cosine", "euclidean", "dot_product"] = "cosine"):
        """
        Initializes the SimilarityScorer.

        Args:
            strategy: The similarity metric to use.
        """
        if strategy not in ["cosine", "euclidean", "dot_product"]:
            raise ValueError(f"Unsupported similarity strategy: {strategy}")
        self.strategy = strategy
        self.use_faiss = config.RECOMMENDER_CONFIG.get("use_faiss", False) and FAISS_AVAILABLE

        if self.use_faiss:
            logger.info("Using FAISS for similarity search.")
        else:
            logger.info("Using scikit-learn for similarity search.")

    def score(self, query_vector: np.ndarray, item_vectors: np.ndarray) -> np.ndarray:
        """
        Calculates the similarity scores between a query vector and a set of item vectors.

        Args:
            query_vector: The vector of the user profile or query item.
            item_vectors: A matrix of item vectors.

        Returns:
            An array of similarity scores.
        """
        if self.use_faiss:
            return self._score_faiss(query_vector, item_vectors)
        else:
            return self._score_sklearn(query_vector, item_vectors)

    def _score_sklearn(self, query_vector: np.ndarray, item_vectors: np.ndarray) -> np.ndarray:
        """
        Calculates similarity using scikit-learn.
        """
        query_vector = query_vector.reshape(1, -1)
        if self.strategy == "cosine":
            return cosine_similarity(query_vector, item_vectors).flatten()
        elif self.strategy == "euclidean":
            # Higher is better, so we invert the distance
            return -euclidean_distances(query_vector, item_vectors).flatten()
        elif self.strategy == "dot_product":
            return np.dot(query_vector, item_vectors.T).flatten()

    def _score_faiss(self, query_vector: np.ndarray, item_vectors: np.ndarray) -> np.ndarray:
        """
        Calculates similarity using FAISS for the top k items.
        """
        d = item_vectors.shape[1]
        index = faiss.IndexFlatL2(d)  # L2 distance is Euclidean
        if self.strategy == "cosine":
            # For cosine similarity, normalize vectors and use inner product
            faiss.normalize_L2(item_vectors)
            index = faiss.IndexFlatIP(d)
        index.add(item_vectors)

        query_vector = query_vector.reshape(1, -1)
        if self.strategy == "cosine":
            faiss.normalize_L2(query_vector)

        # We search for all items, so k = number of items
        k = item_vectors.shape[0]
        distances, indices = index.search(query_vector, k)

        # Re-order scores to match original item order
        scores = np.full(k, -np.inf, dtype=np.float32)
        scores[indices.flatten()] = distances.flatten()

        if self.strategy == "euclidean":
            return -scores # Invert distance
        return scores
