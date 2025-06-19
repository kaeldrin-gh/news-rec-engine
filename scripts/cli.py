"""
Command-line interface for the recommendation engine.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from newsrec.data.loader import download_ag_news
from newsrec.recommender import ContentRecommender
from newsrec.text.vectorizers.base import VectorizerBase
from newsrec.text.vectorizers.sbert import SBertVectorizer
from newsrec.text.vectorizers.tfidf import TfidfVectorizer
from newsrec.text.vectorizers.word2vec import Word2VecVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def train(
    dataset_path: Path = typer.Option(
        ..., "--dataset", "-d", help="Path to the training dataset (CSV or JSON)."
    ),
    vectorizer_name: str = typer.Option(
        "sbert", "--vectorizer", "-v", help="Vectorizer to use (tfidf, word2vec, sbert)."
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save the trained model."
    ),
):
    """
    Trains a recommendation model.
    """
    logger.info(f"Starting training with vectorizer: {vectorizer_name}")

    if str(dataset_path) == "ag_news":
        dataset_path = download_ag_news(Path("data"))

    if vectorizer_name == "tfidf":
        vectorizer = TfidfVectorizer()
    elif vectorizer_name == "word2vec":
        vectorizer = Word2VecVectorizer()
    elif vectorizer_name == "sbert":
        vectorizer = SBertVectorizer()
    else:
        logger.error(f"Invalid vectorizer: {vectorizer_name}")
        raise typer.Exit(code=1)

    from newsrec.data.loader import DataLoader
    loader = DataLoader(filepath=str(dataset_path), text_cols=["title", "description"])
    df = loader.load()

    recommender = ContentRecommender(vectorizer)
    recommender.fit(df, text_field="text")

    output_dir.mkdir(parents=True, exist_ok=True)
    recommender.vectorizer.save(output_dir / "vectorizer.joblib")
    np.save(output_dir / "item_vectors.npy", recommender.item_vectors)
    df[["id"]].to_csv(output_dir / "item_ids.csv", index=False)

    logger.info(f"Model trained and saved to {output_dir}")

@app.command()
def recommend(
    model_dir: Path = typer.Option(
        ..., "--model-dir", "-m", help="Directory where the trained model is saved."
    ),
    user_id: int = typer.Option(..., "--user-id", "-u", help="The user ID to get recommendations for."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of recommendations to return."),
):
    """
    Gets recommendations for a user.
    """
    logger.info(f"Loading model from {model_dir}")
    vectorizer = VectorizerBase.load(model_dir / "vectorizer.joblib")
    item_vectors = np.load(model_dir / "item_vectors.npy")
    item_ids_df = pd.read_csv(model_dir / "item_ids.csv")
    item_ids = item_ids_df["id"].tolist()

    recommender = ContentRecommender(vectorizer)
    recommender.item_vectors = item_vectors
    recommender.item_ids = item_ids

    # This is a placeholder for a real user history lookup
    user_history = item_ids[:5] 

    recommendations = recommender.recommend(user_history, k=top_k)
    print(f"Top {top_k} recommendations for user {user_id}:")
    for item_id, score in recommendations:
        print(f"  - Item ID: {item_id}, Score: {score:.4f}")

if __name__ == "__main__":
    app()
