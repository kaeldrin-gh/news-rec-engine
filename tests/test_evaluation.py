"""
Tests for the evaluation metrics.
"""
from newsrec.evaluation import (
    precision_at_k,
    recall_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
)

def test_precision_at_k():
    assert precision_at_k([1, 2, 3], [1, 4], k=3) == 1 / 3

def test_recall_at_k():
    assert recall_at_k([1, 2, 3], [1, 4], k=3) == 1 / 2

def test_mean_average_precision():
    assert mean_average_precision([1, 2, 3], [1, 3]) == (1 / 1 + 2 / 3) / 2

def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([2, 1, 3], [1, 3]) == 1 / 2
