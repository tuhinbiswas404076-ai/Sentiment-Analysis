"""Prediction and Model Artifact Loader Module for Sentiment Analysis.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
FALLBACK_MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "trained_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"


def _resolve_artifact_path(primary: Path, fallback_filename: str) -> Path:
    if primary.exists():
        return primary
    fallback = FALLBACK_MODEL_DIR / fallback_filename
    if fallback.exists():
        return fallback
    root_fallback = BASE_DIR / fallback_filename
    if root_fallback.exists():
        return root_fallback
    raise FileNotFoundError(f"Model artifact missing: {primary.name}")


def load_model_artifacts() -> Tuple[LogisticRegression, TfidfVectorizer, LabelEncoder]:
    """Load Logistic Regression model, TF-IDF Vectorizer, and Label Encoder."""
    model_file = _resolve_artifact_path(MODEL_PATH, "logistic_regression_model.pkl")
    vec_file = _resolve_artifact_path(VECTORIZER_PATH, "tfidf_vectorizer.pkl")
    enc_file = _resolve_artifact_path(LABEL_ENCODER_PATH, "label_encoder.pkl")

    model = joblib.load(model_file)
    vectorizer = joblib.load(vec_file)
    encoder = joblib.load(enc_file)
    return model, vectorizer, encoder


class SentimentPredictor:
    """Production sentiment prediction service."""

    def __init__(self) -> None:
        self.model, self.vectorizer, self.encoder = load_model_artifacts()

    def predict(self, text: str) -> Dict[str, Any]:
        """Analyze text and return prediction dictionary with error handling."""
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "Neutral / Unknown",
                "confidence": 0.0,
                "probabilities": {"Positive": 0.5, "Negative": 0.5},
                "cleaned_tokens": "",
                "status": "warning",
                "message": "Please enter a valid text sentence to analyze."
            }

        # Truncate extremely long input (>5,000 chars) for performance safety
        if len(text) > 5000:
            text = text[:5000]

        cleaned = preprocess_text(text)
        if not cleaned:
            return {
                "sentiment": "Neutral",
                "confidence": 50.0,
                "probabilities": {"Positive": 0.5, "Negative": 0.5},
                "cleaned_tokens": "(No valid English words remaining after stopword filtering)",
                "status": "info",
                "message": "Text contained only stopwords, numbers, or noise."
            }

        features = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(features)[0]
        pred_idx = self.model.predict(features)[0]

        sentiment_raw = str(self.encoder.inverse_transform([pred_idx])[0]).capitalize()
        confidence = float(max(probs)) * 100

        class_probs = {
            str(cls_name).capitalize(): float(prob)
            for cls_name, prob in zip(self.encoder.classes_, probs)
        }

        return {
            "sentiment": sentiment_raw,
            "confidence": round(confidence, 2),
            "probabilities": class_probs,
            "cleaned_tokens": cleaned,
            "status": "success",
            "message": f"Successfully classified as {sentiment_raw} with {confidence:.2f}% confidence."
        }
