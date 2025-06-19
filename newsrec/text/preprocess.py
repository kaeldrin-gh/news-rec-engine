"""
Text preprocessing pipeline.
"""
import logging
import re
import unicodedata
from typing import List

import spacy
from bs4 import BeautifulSoup
from spacy.tokens import Doc

from newsrec import config

# Configure logging
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load(config.TEXT_PROCESSING_CONFIG["spacy_model"])
except OSError:
    logger.warning(
        f"Spacy model '{config.TEXT_PROCESSING_CONFIG['spacy_model']}' not found. "
        f"Downloading now..."
    )
    spacy.cli.download(config.TEXT_PROCESSING_CONFIG["spacy_model"])
    nlp = spacy.load(config.TEXT_PROCESSING_CONFIG["spacy_model"])


def preprocess_text(text: str, conf: dict = config.TEXT_PROCESSING_CONFIG) -> str:
    """
    Applies a series of preprocessing steps to the input text based on the provided configuration.

    Args:
        text: The raw text to be processed.
        conf: A configuration dictionary that controls the preprocessing steps.

    Returns:
        The processed text as a single string.
    """
    if conf.get("strip_html", True):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

    if conf.get("normalize", True):
        text = unicodedata.normalize("NFKC", text)

    if conf.get("lowercase", True):
        text = text.lower()

    if conf.get("remove_punctuation", True):
        text = re.sub(r"[^\w\s]", "", text)

    if conf.get("remove_digits", True):
        text = re.sub(r"\d", "", text)

    doc: Doc = nlp(text)

    tokens: List[str] = []
    for token in doc:
        is_stop = token.is_stop if conf.get("remove_stopwords", True) else False
        if not is_stop and not token.is_punct and not token.is_space:
            if conf.get("lemmatize", True):
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

    return " ".join(tokens)
