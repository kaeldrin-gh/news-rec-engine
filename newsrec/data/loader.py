"""
Data loading utilities.
"""
import logging
import os
import pandas as pd
from pathlib import Path
import requests
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads data from a CSV or JSON file.
    """

    def __init__(self, filepath: str, text_cols: List[str], id_col: str = "id"):
        """
        Initializes the DataLoader.

        Args:
            filepath: Path to the data file.
            text_cols: A list of columns to be concatenated into a single text field.
            id_col: The name of the column to be used as the item identifier.
        """
        self.filepath = Path(filepath)
        self.text_cols = text_cols
        self.id_col = id_col

    def load(self) -> pd.DataFrame:
        """
        Loads the data from the specified file.

        Returns:
            A pandas DataFrame with the loaded data.
        """
        logger.info(f"Loading data from {self.filepath}...")
        if self.filepath.suffix == ".csv":
            df = pd.read_csv(self.filepath)
        elif self.filepath.suffix == ".json":
            df = pd.read_json(self.filepath, lines=True)
        else:
            raise ValueError(f"Unsupported file type: {self.filepath.suffix}")

        # Combine text columns
        df["text"] = df[self.text_cols].apply(lambda x: " ".join(x.astype(str)), axis=1)
        logger.info("Data loaded successfully.")
        return df

def download_ag_news(data_dir: Path) -> Path:
    """
    Downloads and caches the AG News dataset.

    Args:
        data_dir: The directory where the data should be stored.

    Returns:
        The path to the downloaded dataset.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        logger.info("Downloading AG News dataset...")
        for url, path in [
            ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv", train_path),
            ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv", test_path),
        ]:
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "w", encoding="utf-8") as f:
                f.write(response.text)
        logger.info("Download complete.")

    # Combine train and test sets for this project
    if not (data_dir / "ag_news.csv").exists():
        train_df = pd.read_csv(train_path, header=None, names=["class_index", "title", "description"])
        test_df = pd.read_csv(test_path, header=None, names=["class_index", "title", "description"])
        df = pd.concat([train_df, test_df], ignore_index=True)
        df["id"] = df.index
        df.to_csv(data_dir / "ag_news.csv", index=False)

    return data_dir / "ag_news.csv"
