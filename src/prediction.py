"""Prediction, Batch Processing, and Explainable AI Feature Inspection Module for Sentiment Analysis.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
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


def load_model_artifacts() -> Tuple[Any, TfidfVectorizer, LabelEncoder]:
    """Load Logistic Regression model, TF-IDF Vectorizer, and Label Encoder."""
    model_file = _resolve_artifact_path(MODEL_PATH, "logistic_regression_model.pkl")
    vec_file = _resolve_artifact_path(VECTORIZER_PATH, "tfidf_vectorizer.pkl")
    enc_file = _resolve_artifact_path(LABEL_ENCODER_PATH, "label_encoder.pkl")

    model = joblib.load(model_file)
    vectorizer = joblib.load(vec_file)
    encoder = joblib.load(enc_file)
    return model, vectorizer, encoder


class SentimentPredictor:
    """Production sentiment prediction & Explainable AI service."""

    def __init__(self) -> None:
        self.model, self.vectorizer, self.encoder = load_model_artifacts()
        # Precompute vocabulary index mapping and model coefficients
        self.vocab = self.vectorizer.vocabulary_
        self.coefs = self._extract_coefs()

    def _extract_coefs(self) -> np.ndarray:
        """Extract linear coefficients safely from standard or calibrated models."""
        try:
            if hasattr(self.model, "coef_"):
                return self.model.coef_[0]
            if hasattr(self.model, "calibrated_classifiers_") and self.model.calibrated_classifiers_:
                sub_coefs = [
                    c.estimator.coef_[0] for c in self.model.calibrated_classifiers_
                    if hasattr(c, "estimator") and hasattr(c.estimator, "coef_")
                ]
                if sub_coefs:
                    return np.mean(sub_coefs, axis=0)
            if hasattr(self.model, "estimator") and hasattr(self.model.estimator, "coef_"):
                return self.model.estimator.coef_[0]
        except Exception:
            pass
        # Fallback dummy zero array matching vocabulary length
        return np.zeros(len(self.vocab))

    def predict(self, text: str) -> Dict[str, Any]:
        """Analyze single text and return prediction dictionary with error handling."""
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "Neutral / Unknown",
                "confidence": 0.0,
                "probabilities": {"Positive": 0.5, "Negative": 0.5},
                "cleaned_tokens": "",
                "highlighted_html": "<em>Please enter a sentence to analyze.</em>",
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
                "highlighted_html": f"<span>{text}</span>",
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

        # Explainable AI: Generate Keyword Highlighted HTML
        highlighted_html = self.explain_text(text)

        return {
            "sentiment": sentiment_raw,
            "confidence": round(confidence, 2),
            "probabilities": class_probs,
            "cleaned_tokens": cleaned,
            "highlighted_html": highlighted_html,
            "status": "success",
            "message": f"Successfully classified as {sentiment_raw} with {confidence:.2f}% confidence."
        }

    def explain_text(self, text: str) -> str:
        """Generate HTML with green pills for positive words and red pills for negative words."""
        words = re.findall(r"\b\w+\b|\S+", text)
        highlighted_words = []

        for w in words:
            clean_w = w.lower()
            if clean_w in self.vocab:
                idx = self.vocab[clean_w]
                score = float(self.coefs[idx])
                if score > 0.4:
                    # Positive Word -> Emerald Green Pill
                    pill = (
                        f'<span style="background-color: rgba(16, 185, 129, 0.25); '
                        f'color: #10b981; padding: 2px 6px; border-radius: 4px; '
                        f'font-weight: 600;" title="Positive Impact: +{score:.2f}">{w}</span>'
                    )
                elif score < -0.4:
                    # Negative Word -> Crimson Red Pill
                    pill = (
                        f'<span style="background-color: rgba(239, 68, 68, 0.25); '
                        f'color: #ef4444; padding: 2px 6px; border-radius: 4px; '
                        f'font-weight: 600;" title="Negative Impact: {score:.2f}">{w}</span>'
                    )
                else:
                    pill = w
            else:
                pill = w
            highlighted_words.append(pill)

        return (
            '<div style="line-height: 1.8; font-size: 1.05rem; padding: 12px; '
            'border-radius: 8px; background: #1e293b; border: 1px solid #334155;">'
            + " ".join(highlighted_words) +
            '</div>'
        )

    def predict_batch(self, file_path_or_df: Any, text_column: str = "") -> Tuple[pd.DataFrame, Dict[str, int], str]:
        """Process CSV / Excel batch dataset, append predictions, and return (df, summary_dict, temp_file_path)."""
        if isinstance(file_path_or_df, str) or isinstance(file_path_or_df, Path):
            file_str = str(file_path_or_df)
            if file_str.endswith(".xlsx") or file_str.endswith(".xls"):
                df = pd.read_excel(file_str)
            else:
                df = pd.read_csv(file_str)
        elif isinstance(file_path_or_df, pd.DataFrame):
            df = file_path_or_df.copy()
        else:
            raise ValueError("Input must be a file path string or pandas DataFrame.")

        if not text_column:
            # Auto-detect text column if not specified
            text_candidates = [c for c in df.columns if any(k in c.lower() for k in ["text", "review", "tweet", "comment", "content", "sentence"])]
            text_column = text_candidates[0] if text_candidates else df.columns[0]

        texts = df[text_column].astype(str).tolist()
        cleaned_list = [preprocess_text(t) for t in texts]
        features = self.vectorizer.transform(cleaned_list)
        probs_matrix = self.model.predict_proba(features)
        pred_indices = self.model.predict(features)
        sentiments = [str(s).capitalize() for s in self.encoder.inverse_transform(pred_indices)]
        confidences = [round(float(max(p)) * 100, 2) for p in probs_matrix]

        df["Predicted_Sentiment"] = sentiments
        df["Confidence_%"] = confidences
        df["Cleaned_Tokens"] = cleaned_list

        counts = df["Predicted_Sentiment"].value_counts().to_dict()

        # Save output to temp CSV file for user download
        temp_dir = tempfile.gettempdir()
        out_path = os.path.join(temp_dir, "sentiment_predictions_annotated.csv")
        df.to_csv(out_path, index=False)

        return df, counts, out_path

    def get_top_features(self, top_n: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return top N positive and top N negative vocabulary terms as DataFrames."""
        feature_names = self.vectorizer.get_feature_names_out()
        sorted_indices = np.argsort(self.coefs)

        top_negative = pd.DataFrame({
            "Negative Keyword": [feature_names[i] for i in sorted_indices[:top_n]],
            "Weight (Impact)": [round(float(self.coefs[i]), 3) for i in sorted_indices[:top_n]]
        })

        top_positive = pd.DataFrame({
            "Positive Keyword": [feature_names[i] for i in sorted_indices[-top_n:][::-1]],
            "Weight (Impact)": [round(float(self.coefs[i]), 3) for i in sorted_indices[-top_n:][::-1]]
        })

        return top_positive, top_negative
