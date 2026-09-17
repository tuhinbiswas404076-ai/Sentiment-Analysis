"""Gradio Web Application for AI Sentiment Analysis.
Production-ready entry point for Hugging Face Spaces & GitHub repository.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.prediction import SentimentPredictor

# Initialize Sentiment Predictor Service
try:
    predictor = SentimentPredictor()
    INIT_ERROR = None
except Exception as exc:
    predictor = None
    INIT_ERROR = str(exc)


def analyze_text(input_text: str) -> Tuple[str, float, Dict[str, float], str]:
    """Gradio prediction handler returning (sentiment_markdown, confidence_num, probabilities_dict, cleaned_tokens)."""
    if INIT_ERROR or predictor is None:
        return (
            f"❌ **System Error:** Model failed to initialize: {INIT_ERROR}",
            0.0,
            {"Positive": 0.0, "Negative": 0.0},
            "(Error)"
        )

    result = predictor.predict(input_text)
    sentiment = result["sentiment"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    cleaned = result["cleaned_tokens"]
    status = result["status"]
    msg = result["message"]

    if status != "success":
        return (
            f"⚠️ **Notice:** {msg}",
            confidence,
            probs,
            cleaned
        )

    emoji = "🟢" if sentiment.lower() == "positive" else "🔴"
    sentiment_md = f"### {emoji} **Predicted Sentiment:** {sentiment}\n**Confidence:** {confidence:.2f}%"

    return sentiment_md, confidence, probs, cleaned


# Custom CSS styling for modern UI
custom_css = """
.container { max-width: 900px; margin: auto; }
.title-box { text-align: center; margin-bottom: 1rem; }
.sentiment-output { padding: 15px; border-radius: 8px; font-size: 1.2rem; }
"""

# Build Gradio Blocks Interface
with gr.Blocks(title="AI Sentiment Analysis") as demo:
    gr.Markdown(
        """
        # 🎭 AI Sentiment Analysis
        ### Machine Learning Based Text Sentiment Classification
        Enter any sentence, review, or tweet below to evaluate its sentiment in real-time.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                lines=5,
                placeholder="Type your review or text here (e.g. 'I absolutely loved this movie. It was amazing!')...",
                label="Input Sentence / Review",
                elem_id="input-text-area"
            )
            submit_btn = gr.Button("⚡ Analyze Sentiment", variant="primary")

        with gr.Column(scale=2):
            sentiment_output = gr.Markdown(
                value="*Submit text to view sentiment prediction*",
                label="Prediction Result"
            )
            confidence_output = gr.Number(label="Confidence Score (%)", precision=2)
            probabilities_output = gr.Label(label="Class Probability Distribution", num_top_classes=2)
            tokens_output = gr.Textbox(label="Cleaned & Lemmatized Tokens", interactive=False)

    submit_btn.click(
        fn=analyze_text,
        inputs=[input_box],
        outputs=[sentiment_output, confidence_output, probabilities_output, tokens_output]
    )

    gr.Examples(
        examples=[
            ["I absolutely loved this movie. It was amazing!"],
            ["Terrible experience. The service was slow and workers were rude."],
            ["The plot was okay, but the acting was somewhat weak."],
            ["Great quality product, fast shipping, and awesome support!"]
        ],
        inputs=[input_box],
        outputs=[sentiment_output, confidence_output, probabilities_output, tokens_output],
        fn=analyze_text,
        cache_examples=False
    )

    gr.Markdown(
        """
        ---
        **Model Specs:** TF-IDF (150k features) + Logistic Regression (L2 Regularized) | Trained on 88,000+ multi-domain dataset samples.
        """
    )

if __name__ == "__main__":
    demo.launch(css=custom_css)

