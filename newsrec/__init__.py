"""
NewsRec: A Content-Based News Recommendation Engine

A modular, production-quality recommendation engine for news articles.
"""

__version__ = "0.1.0"
__author__ = "NewsRec Team"

from newsrec.recommender import ContentRecommender
from newsrec.data.loader import DataLoader

__all__ = ["ContentRecommender", "DataLoader"]
