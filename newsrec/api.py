"""
FastAPI service for the recommendation engine.
"""
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from newsrec.recommender import ContentRecommender
from newsrec.text.vectorizers.base import VectorizerBase

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Recommendation API",
    description="A content-based recommendation engine for news articles.",
    version="0.1.0",
)

# --- Globals ---
recommender: ContentRecommender = None

class UserHistory(BaseModel):
    user_history: List[int]
    k: int = 10

@app.on_event("startup")
def load_model():
    """
    Loads the trained recommender model on startup.
    """
    global recommender
    model_path = Path("models/sbert/vectorizer.joblib") # Default model path
    if not model_path.exists():
        logger.warning("Model not found. API will not be able to make recommendations.")
        return

    try:
        vectorizer = VectorizerBase.load(model_path)
        recommender = ContentRecommender(vectorizer)
        # Here you would typically load a pre-fit recommender with item vectors
        # For simplicity, we are not doing that in this example.
        logger.info("Recommender loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Error loading model.")

@app.post("/recommend", summary="Get news recommendations")
def get_recommendations(history: UserHistory):
    """
    Provides a list of recommended article IDs based on user history.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not available.")

    try:
        recommendations = recommender.recommend(history.user_history, k=history.k)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations.")
