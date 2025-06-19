"""
Central configuration for the news recommendation engine.
"""
from pathlib import Path
from typing import Any, Dict, List

# --- Project Paths ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# --- Text Preprocessing ---
TEXT_PROCESSING_CONFIG: Dict[str, Any] = {
    "normalize": True,
    "strip_html": True,
    "remove_punctuation": True,
    "remove_digits": True,
    "lowercase": True,
    "lemmatize": True,
    "remove_stopwords": True,
    "spacy_model": "en_core_web_sm",
}

# --- Vectorizer Settings ---
VECTORIZER_CONFIG: Dict[str, Any] = {
    "tfidf": {
        "max_features": 5000,
    },
    "word2vec": {
        "vector_size": 300,
        "window": 5,
        "min_count": 2,
        "workers": 4,
        "sg": 1,  # 1 for skip-gram, 0 for CBOW
        "pretrained_path": "GoogleNews-vectors-negative300.bin", # Optional
    },
    "sbert": {
        "model_name": "all-MiniLM-L6-v2",
    },
}

# --- Recommender Settings ---
RECOMMENDER_CONFIG: Dict[str, Any] = {
    "user_profile_strategy": "mean",  # "mean" or "weighted_average"
    "user_profile_weighted_alpha": 0.1, # Decay factor for weighted average
    "use_faiss": False, # Set to True to use FAISS for similarity search
}

# --- Evaluation Settings ---
EVALUATION_CONFIG: Dict[str, Any] = {
    "test_split_ratio": 0.2,
    "metrics_k_values": [5, 10],
    "history_length_for_eval": 5,
}

# --- Reproducibility ---
RANDOM_SEED: int = 42

# --- API Settings ---
API_CONFIG: Dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8000,
}
