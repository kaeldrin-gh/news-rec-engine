"""
Base class for vectorizers.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self
import numpy as np
import joblib

class VectorizerBase(ABC):
    """
    Abstract base class for all vectorizer implementations.
    """

    @abstractmethod
    def fit(self, docs: list[str]) -> Self:
        """
        Fits the vectorizer on a list of documents.

        Args:
            docs: A list of text documents.

        Returns:
            The fitted vectorizer instance.
        """
        pass

    @abstractmethod
    def transform(self, docs: list[str]) -> np.ndarray:
        """
        Transforms a list of documents into their vector representations.

        Args:
            docs: A list of text documents.

        Returns:
            A NumPy array of document vectors.
        """
        pass

    def save(self, path: Path) -> None:
        """
        Saves the fitted vectorizer to a file.

        Args:
            path: The path to save the vectorizer to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "VectorizerBase":
        """
        Loads a vectorizer from a file.

        Args:
            path: The path to load the vectorizer from.

        Returns:
            An instance of the loaded vectorizer.
        """
        return joblib.load(path)
