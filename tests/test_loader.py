"""
Tests for the data loader.
"""
import pandas as pd
from pathlib import Path
from newsrec.data.loader import DataLoader, download_ag_news

def test_download_ag_news(tmp_path: Path):
    """
    Tests the AG News download helper.
    """
    ag_news_path = download_ag_news(tmp_path)
    assert ag_news_path.exists()
    df = pd.read_csv(ag_news_path)
    assert "id" in df.columns
    assert "title" in df.columns
    assert "description" in df.columns

def test_data_loader_csv(tmp_path: Path):
    """
    Tests the DataLoader with a CSV file.
    """
    csv_content = "id,title,description\n1,title1,desc1\n2,title2,desc2"
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(csv_content)

    loader = DataLoader(filepath=str(csv_path), text_cols=["title", "description"])
    df = loader.load()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "text" in df.columns
    assert df.loc[0, "text"] == "title1 desc1"
