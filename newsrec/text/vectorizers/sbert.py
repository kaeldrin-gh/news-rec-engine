"""
Sentence-BERT Vectorizer implementation.
"""
import logging
from typing import List, Self

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from newsrec.text.vectorizers.base import VectorizerBase
from newsrec import config

# Configure logging
logger = logging.getLogger(__name__)

class SBertVectorizer(VectorizerBase):
    """
    A wrapper around the sentence-transformers library.
    """

    def __init__(self, model_name: str = config.VECTORIZER_CONFIG["sbert"]["model_name"]):
        """
        Initializes the SBertVectorizer.

        Args:
            model_name: The name of the sentence-transformer model to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def fit(self, docs: List[str]) -> Self:
        """
        This method is not needed for pre-trained sentence-transformers,
        but is included for compatibility with the base class.

        Args:
            docs: A list of documents.

        Returns:
            The vectorizer instance.
        """
        logger.info("SBertVectorizer does not require fitting. Model is already trained.")
        return self

    def transform(self, docs: List[str]) -> np.ndarray:
        """
        Transforms the documents into sentence embeddings.

        Args:
            docs: A list of documents.

        Returns:
            A numpy array of the document vectors.
        """
        return self.model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
