"""
Tests for the vectorizers.
"""
import numpy as np
import pytest
from newsrec.text.vectorizers.sbert import SBertVectorizer
from newsrec.text.vectorizers.tfidf import TfidfVectorizer


@pytest.fixture
def sample_docs():
    return ["this is the first document", "this document is the second document"]

def test_tfidf_vectorizer(sample_docs):
    """
    Tests the TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(sample_docs)
    vectors = vectorizer.transform(sample_docs)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == len(sample_docs)

def test_sbert_vectorizer(sample_docs):
    """
    Tests the SBertVectorizer.
    """
    vectorizer = SBertVectorizer()
    vectors = vectorizer.transform(sample_docs)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == len(sample_docs)
    assert vectors.shape[1] == 384  # all-MiniLM-L6-v2 embedding dimension
