"""
Tests for the recommendation engine.
"""
import pandas as pd
import pytest
from newsrec.recommender import ContentRecommender
from newsrec.text.vectorizers.sbert import SBertVectorizer


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "content": [
                "science and technology",
                "politics and government",
                "health and wellness",
                "space exploration",
                "political news",
            ],
        }
    )


def test_recommender(sample_data):
    """
    Tests the ContentRecommender.
    """
    vectorizer = SBertVectorizer()
    recommender = ContentRecommender(vectorizer)
    recommender.fit(sample_data, text_field="content")

    user_history = [2, 5]  # History of politics
    recommendations = recommender.recommend(user_history, k=2)

    assert len(recommendations) == 2
    # The top recommendation should not be from the user's history
    assert recommendations[0][0] not in user_history
