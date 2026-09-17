"""Text Preprocessing Pipeline for Sentiment Analysis.

Includes HTML cleaning, URL/mention removal, contraction expansion, tokenization,
negation-preserved stopword removal, and lemmatization.
"""
from __future__ import annotations

import re
import string
import html
import nltk
from contractions import fix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_data() -> None:
    """Download required NLTK datasets silently if missing."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(
                f"tokenizers/{resource}"
                if resource.startswith("punkt")
                else f"corpora/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


# Run NLTK download check upon import
ensure_nltk_data()

RAW_STOP_WORDS = set(stopwords.words("english"))
NEGATIONS = {
    "no", "not", "nor", "neither", "never", "none", "cannot",
    "don't", "aren't", "couldn't", "didn't", "doesn't", "hadn't",
    "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't",
    "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"
}
STOP_WORDS = RAW_STOP_WORDS - NEGATIONS
LEMMATIZER = WordNetLemmatizer()


def _normalize_text(text: str) -> str:
    """Clean HTML tags, Twitter mentions, URLs, and excess whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("<br />", " ").replace("<br>", " ").replace("<br/>", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """Apply NLP preprocessing pipeline:
    Contraction expansion, noise cleaning, tokenization, stopword filtering with negation preservation, and lemmatization.
    """
    clean_text = _normalize_text(text).lower()
    clean_text = fix(clean_text)
    clean_text = re.sub(r"[^a-zA-Z\s]", " ", clean_text)
    tokens = word_tokenize(clean_text)
    tokens = [
        token for token in tokens
        if token not in STOP_WORDS and token not in string.punctuation and len(token) > 1
    ]
    lemmatized_tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)
