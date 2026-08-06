"""Advanced Flask REST API for sentiment analysis."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import BadRequest, HTTPException

from utils.preprocessing import (
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
    load_model_artifacts,
    preprocess_text,
)

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "sentiment_history.db")))

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.update(
    JSON_SORT_KEYS=False,
    MAX_CONTENT_LENGTH=int(os.getenv("MAX_CONTENT_LENGTH", "1048576")),
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-key"),
)

CORS(
    app,
    resources={r"/*": {"origins": os.getenv("CORS_ORIGINS", "http://localhost:5000").split(",")}},
)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[os.getenv("RATE_LIMIT", "120 per minute")],
    storage_uri="memory://",
)

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

MODEL, VECTORIZER, LABEL_ENCODER = load_model_artifacts(
    model_path=MODEL_PATH,
    vectorizer_path=VECTORIZER_PATH,
    label_encoder_path=LABEL_ENCODER_PATH,
)

SENTIMENT_CLASSES = [label for label in LABEL_ENCODER.classes_]
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_LIMIT = 256


def init_db() -> None:
    """Create the SQLite history table if it does not already exist."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )
        conn.commit()


init_db()


@app.after_request
def add_security_headers(response):
    """Add lightweight security headers for the web and API responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:"
    return response


@app.errorhandler(400)
def bad_request_error(error):
    if request.path in {"/health", "/predict", "/predict/batch", "/history", "/analytics", "/model-info"}:
        return jsonify({"success": False, "error": "Invalid request", "message": "The request payload is invalid."}), 400
    return jsonify({"success": False, "error": "Bad request", "message": "The request could not be processed."}), 400


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"success": False, "error": "Not found", "message": "The requested endpoint does not exist."}), 404


@app.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify({"success": False, "error": "Method not allowed", "message": "The requested method is not allowed for this endpoint."}), 405


@app.errorhandler(413)
def payload_too_large_error(error):
    return jsonify({"success": False, "error": "Payload too large", "message": "The submitted payload exceeds the maximum allowed size."}), 413


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Unhandled application error")
    return jsonify({"success": False, "error": "Internal server error", "message": "An unexpected error occurred."}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "success": True,
        "status": "ok",
        "model_loaded": True,
        "vectorizer_loaded": True,
        "label_encoder_loaded": True,
    })


@app.route("/predict", methods=["POST"])
@limiter.limit("20 per minute")
def predict():
    start_time = time.perf_counter()
    try:
        payload = request.get_json(silent=False)
    except BadRequest:
        return jsonify({"success": False, "error": "Invalid JSON", "message": "The request body must be valid JSON."}), 400

    if not isinstance(payload, dict):
        return jsonify({"success": False, "error": "Invalid request", "message": "A JSON object is required."}), 400

    text = payload.get("text")
    if text is None:
        return jsonify({"success": False, "error": "Missing text", "message": "The 'text' field is required."}), 400
    if not isinstance(text, str):
        return jsonify({"success": False, "error": "Invalid text", "message": "The 'text' field must be a string."}), 400
    if not text.strip():
        return jsonify({"success": False, "error": "Empty text", "message": "The 'text' field cannot be empty."}), 400
    if len(text) > 5000:
        return jsonify({"success": False, "error": "Payload too large", "message": "The input text is too long."}), 413

    cleaned_text = preprocess_text(text)
    cache_key = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()
    if cache_key in CACHE:
        result = CACHE[cache_key].copy()
        result["cached"] = True
    else:
        features = VECTORIZER.transform([cleaned_text])
        probabilities = MODEL.predict_proba(features)[0]
        predicted_index = int(MODEL.predict(features)[0])
        sentiment = LABEL_ENCODER.inverse_transform([predicted_index])[0]
        probs_dict = {
            label: round(float(prob), 6)
            for label, prob in zip(LABEL_ENCODER.classes_, probabilities)
        }
        confidence = round(float(max(probabilities)), 6)
        result = {
            "success": True,
            "text": text,
            "sentiment": str(sentiment),
            "confidence": confidence,
            "probabilities": probs_dict,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }
        CACHE[cache_key] = result
        if len(CACHE) > CACHE_LIMIT:
            CACHE.pop(next(iter(CACHE)))

    result["processing_time_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
    store_prediction(text, result["sentiment"], result["confidence"])
    app.logger.info("Prediction completed for request")
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
@limiter.limit("30 per minute")
def predict_batch():
    start_time = time.perf_counter()
    try:
        payload = request.get_json(silent=False)
    except BadRequest:
        return jsonify({"success": False, "error": "Invalid JSON", "message": "The request body must be valid JSON."}), 400

    if not isinstance(payload, dict) or "texts" not in payload:
        return jsonify({"success": False, "error": "Invalid request", "message": "A 'texts' list is required."}), 400

    texts = payload.get("texts")
    if not isinstance(texts, list):
        return jsonify({"success": False, "error": "Invalid request", "message": "The 'texts' field must be a list."}), 400
    if not texts:
        return jsonify({"success": False, "error": "Empty batch", "message": "At least one text item is required."}), 400
    if len(texts) > 50:
        return jsonify({"success": False, "error": "Payload too large", "message": "Batch size cannot exceed 50 items."}), 413

    predictions = []
    for item in texts:
        if not isinstance(item, str):
            return jsonify({"success": False, "error": "Invalid batch item", "message": "Each item in 'texts' must be a string."}), 400
        if not item.strip():
            return jsonify({"success": False, "error": "Empty text", "message": "Batch items cannot be empty strings."}), 400
        prediction = predict_single_text(item)
        predictions.append(prediction)

    summary = {
        "total_texts": len(predictions),
        "positive_count": sum(1 for item in predictions if item["sentiment"] == "positive"),
        "negative_count": sum(1 for item in predictions if item["sentiment"] == "negative"),
        "neutral_count": sum(1 for item in predictions if item["sentiment"] == "neutral"),
    }
    return jsonify({
        "success": True,
        "predictions": predictions,
        "summary": summary,
        "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
    })


@app.route("/history", methods=["GET"])
def get_history():
    limit = request.args.get("limit", default="20")
    try:
        limit_value = int(limit)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid limit", "message": "The limit must be an integer."}), 400
    rows = fetch_history(limit_value)
    return jsonify({"success": True, "history": rows})


@app.route("/history", methods=["DELETE"])
def clear_history():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    return jsonify({"success": True, "message": "Prediction history cleared."})


@app.route("/analytics", methods=["GET"])
def analytics():
    rows = fetch_history(100000)
    total = len(rows)
    positive = sum(1 for item in rows if item["sentiment"] == "positive")
    negative = sum(1 for item in rows if item["sentiment"] == "negative")
    neutral = sum(1 for item in rows if item["sentiment"] == "neutral")
    percentages = {
        "positive": round((positive / total) * 100, 2) if total else 0.0,
        "negative": round((negative / total) * 100, 2) if total else 0.0,
        "neutral": round((neutral / total) * 100, 2) if total else 0.0,
    }
    return jsonify({
        "success": True,
        "total_predictions": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "percentages": percentages,
        "average_confidence": round(sum(item["confidence"] for item in rows) / total, 6) if total else 0.0,
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "success": True,
        "model_name": "logistic_regression_model.pkl",
        "model_type": "LogisticRegression",
        "vectorizer_type": "TfidfVectorizer",
        "num_classes": len(SENTIMENT_CLASSES),
        "available_sentiment_classes": SENTIMENT_CLASSES,
    })


@app.route("/swagger.json", methods=["GET"])
def swagger_spec():
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "AI Sentiment Analysis API",
            "version": "1.0.0",
            "description": "REST API for 3-class sentiment classification.",
        },
        "paths": {
            "/": {"get": {"summary": "API homepage"}},
            "/health": {"get": {"summary": "Health status"}},
            "/predict": {"post": {"summary": "Predict a single text"}},
            "/predict/batch": {"post": {"summary": "Predict multiple texts"}},
            "/history": {"get": {"summary": "Get prediction history"}, "delete": {"summary": "Clear history"}},
            "/analytics": {"get": {"summary": "Get analytics"}},
            "/model-info": {"get": {"summary": "Get model metadata"}},
        },
    }
    return jsonify(spec)


@app.route("/docs", methods=["GET"])
def docs():
    return """
    <!DOCTYPE html>
    <html lang='en'>
      <head>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <title>Swagger UI</title>
        <link rel='stylesheet' href='https://unpkg.com/swagger-ui-dist@5.17.2/swagger-ui.css' />
      </head>
      <body>
        <div id='swagger-ui'></div>
        <script src='https://unpkg.com/swagger-ui-dist@5.17.2/swagger-ui-bundle.js'></script>
        <script>
          window.onload = () => {
            SwaggerUIBundle({
              url: '/swagger.json',
              dom_id: '#swagger-ui'
            })
          }
        </script>
      </body>
    </html>
    """


def predict_single_text(text: str) -> Dict[str, Any]:
    cleaned_text = preprocess_text(text)
    features = VECTORIZER.transform([cleaned_text])
    probabilities = MODEL.predict_proba(features)[0]
    predicted_index = int(MODEL.predict(features)[0])
    sentiment = LABEL_ENCODER.inverse_transform([predicted_index])[0]
    probs_dict = {label: round(float(prob), 6) for label, prob in zip(LABEL_ENCODER.classes_, probabilities)}
    confidence = round(float(max(probabilities)), 6)
    return {
        "text": text,
        "sentiment": str(sentiment),
        "confidence": confidence,
        "probabilities": probs_dict,
    }


def store_prediction(input_text: str, sentiment: str, confidence: float) -> None:
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, input_text, sentiment, confidence) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                input_text[:1000],
                sentiment,
                confidence,
            ),
        )
        conn.commit()


def fetch_history(limit_value: int) -> List[Dict[str, Any]]:
    with sqlite3.connect(DATABASE_PATH) as conn:
        rows = conn.execute(
            "SELECT timestamp, input_text, sentiment, confidence FROM predictions ORDER BY id DESC LIMIT ?",
            (limit_value,),
        ).fetchall()
    return [
        {
            "timestamp": row[0],
            "input_text": row[1],
            "sentiment": row[2],
            "confidence": row[3],
        }
        for row in rows
    ]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
