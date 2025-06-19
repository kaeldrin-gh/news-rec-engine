"""
Tests for the text preprocessing pipeline.
"""
from newsrec.text.preprocess import preprocess_text

def test_preprocess_text():
    """
    Tests the full preprocessing pipeline.
    """
    raw_text = "<p>This is a test with <b>HTML</b> tags, 123 numbers, and punctuation!</p>"
    processed_text = preprocess_text(raw_text)
    expected_text = "test html tag number punctuation"
    assert processed_text == expected_text
