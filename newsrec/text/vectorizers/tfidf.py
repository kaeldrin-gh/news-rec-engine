"""
TF-IDF Vectorizer implementation.
"""
from typing import List, Self

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

from newsrec.text.vectorizers.base import VectorizerBase
from newsrec import config

class TfidfVectorizer(VectorizerBase):
    """
    A wrapper around scikit-learn's TfidfVectorizer.
    """

    def __init__(self, max_features: int = config.VECTORIZER_CONFIG["tfidf"]["max_features"]):
        """
        Initializes the TfidfVectorizer.

        Args:
            max_features: The maximum number of features to keep.
        """
        self.vectorizer = SklearnTfidfVectorizer(max_features=max_features)

    def fit(self, docs: List[str]) -> Self:
        """
        Fits the TF-IDF vectorizer to the documents.

        Args:
            docs: A list of documents.

        Returns:
            The fitted vectorizer.
        """
        self.vectorizer.fit(docs)
        return self

    def transform(self, docs: List[str]) -> np.ndarray:
        """
        Transforms the documents into TF-IDF vectors.

        Args:
            docs: A list of documents.

        Returns:
            A numpy array of the document vectors.
        """
        return self.vectorizer.transform(docs).toarray()
