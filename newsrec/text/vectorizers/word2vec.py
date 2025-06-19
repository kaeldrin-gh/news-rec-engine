"""
Word2Vec Vectorizer implementation.
"""
import logging
from typing import List, Self

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from nltk.tokenize import word_tokenize

from newsrec.text.vectorizers.base import VectorizerBase
from newsrec import config

# Configure logging
logger = logging.getLogger(__name__)

class Word2VecVectorizer(VectorizerBase):
    """
    A wrapper around gensim's Word2Vec model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Word2VecVectorizer.

        Args:
            **kwargs: Keyword arguments for the gensim Word2Vec model.
        """
        self.model_params = {**config.VECTORIZER_CONFIG["word2vec"], **kwargs}
        self.model = None
        self.vector_size = self.model_params["vector_size"]

    def fit(self, docs: List[str]) -> Self:
        """
        Fits the Word2Vec model to the documents.

        Args:
            docs: A list of documents.

        Returns:
            The fitted vectorizer.
        """
        tokenized_docs = [word_tokenize(doc) for doc in docs]
        pretrained_path = self.model_params.pop("pretrained_path", None)

        if pretrained_path and KeyedVectors.load_word2vec_format(pretrained_path, binary=True):
            logger.info(f"Loading pre-trained Word2Vec model from {pretrained_path}")
            self.model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        else:
            logger.info("Training Word2Vec model from scratch...")
            self.model = Word2Vec(sentences=tokenized_docs, **self.model_params).wv
        return self

    def transform(self, docs: List[str]) -> np.ndarray:
        """
        Transforms the documents into Word2Vec vectors by averaging word embeddings.

        Args:
            docs: A list of documents.

        Returns:
            A numpy array of the document vectors.
        """
        vectors = []
        for doc in docs:
            tokens = word_tokenize(doc)
            doc_vector = np.mean(
                [self.model[token] for token in tokens if token in self.model],
                axis=0,
            )
            if np.isnan(doc_vector).any():
                vectors.append(np.zeros(self.vector_size))
            else:
                vectors.append(doc_vector)
        return np.array(vectors)
