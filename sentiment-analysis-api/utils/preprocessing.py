"""Text preprocessing utilities compatible with the notebook pipeline."""
from __future__ import annotations

import os
import re
import string
from pathlib import Path
from typing import List, Tuple

import joblib
import nltk
from contractions import fix
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / ".." / "TextAnalytics.txt"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"


def ensure_nltk_data() -> None:
    """Download the small NLP datasets required for the preprocessing pipeline."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "vader_lexicon"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}" if resource == "stopwords" else f"corpora/{resource}" if resource == "wordnet" else f"sentiment/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_data()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
ANALYZER = SentimentIntensityAnalyzer()


def _normalize_text(text: str) -> str:
    """Lowercase and remove visible HTML-like markup from the review text."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    text = text.replace("<br />", " ")
    text = text.replace("<br>", " ")
    text = text.replace("<br/>", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """Apply the notebook-style preprocessing steps to a single text sample."""
    clean_text = _normalize_text(text).lower()
    clean_text = fix(clean_text)
    clean_text = re.sub(r"[^a-zA-Z\s]", " ", clean_text)
    tokens = word_tokenize(clean_text)
    tokens = [token for token in tokens if token not in STOP_WORDS and token not in string.punctuation]
    lemmatized_tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)


def assign_sentiment_label(text: str) -> str:
    """Use VADER compound scores to mirror the notebook's three-class labeling."""
    score = ANALYZER.polarity_scores(text)["compound"]
    if score >= 0.5:
        return "positive"
    if score <= 0.0:
        return "negative"
    return "neutral"


def load_training_data(dataset_path: Path | None = None) -> Tuple[List[str], List[str]]:
    """Load the sentiment dataset and assign labels using the notebook's rule."""
    source_path = dataset_path or DATASET_PATH.resolve()
    texts: List[str] = []
    labels: List[str] = []

    with open(source_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            line = raw_line.strip()
            if "," not in line:
                continue
            _, review_text = line.split(",", 1)
            review_text = review_text.strip().strip('"')
            if not review_text:
                continue
            texts.append(review_text)
            labels.append(assign_sentiment_label(review_text))

    return texts, labels


def train_and_save_artifacts(
    dataset_path: Path | None = None,
    model_path: Path | None = None,
    vectorizer_path: Path | None = None,
    label_encoder_path: Path | None = None,
) -> Tuple[LogisticRegression, TfidfVectorizer, LabelEncoder]:
    """Train the model and persist the artifacts used by the API."""
    model_path = model_path or MODEL_PATH
    vectorizer_path = vectorizer_path or VECTORIZER_PATH
    label_encoder_path = label_encoder_path or LABEL_ENCODER_PATH

    texts, labels = load_training_data(dataset_path)
    processed_texts = [preprocess_text(text) for text in texts]

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    features = vectorizer.fit_transform(processed_texts)

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(features, encoded_labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(encoder, label_encoder_path)
    return model, vectorizer, encoder


def load_model_artifacts(
    model_path: Path | None = None,
    vectorizer_path: Path | None = None,
    label_encoder_path: Path | None = None,
) -> Tuple[LogisticRegression, TfidfVectorizer, LabelEncoder]:
    """Load model artifacts, training them if they do not exist yet."""
    model_path = model_path or MODEL_PATH
    vectorizer_path = vectorizer_path or VECTORIZER_PATH
    label_encoder_path = label_encoder_path or LABEL_ENCODER_PATH

    if not model_path.exists() or not vectorizer_path.exists() or not label_encoder_path.exists():
        return train_and_save_artifacts(model_path=model_path, vectorizer_path=vectorizer_path, label_encoder_path=label_encoder_path)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    encoder = joblib.load(label_encoder_path)
    return model, vectorizer, encoder
