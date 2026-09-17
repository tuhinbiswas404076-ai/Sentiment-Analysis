"""Gradio Multi-Tab Web Application for AI Sentiment Analysis & Explainable AI.
Production-ready entry point for Hugging Face Spaces & GitHub repository.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import pandas as pd

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


def analyze_single_text(input_text: str) -> Tuple[str, float, Dict[str, float], str, str]:
    """Single text analysis handler."""
    if INIT_ERROR or predictor is None:
        return (
            f"❌ **System Error:** Model failed to initialize: {INIT_ERROR}",
            0.0,
            {"Positive": 0.0, "Negative": 0.0},
            "<em>Error loading model.</em>",
            "(Error)"
        )

    result = predictor.predict(input_text)
    sentiment = result["sentiment"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    cleaned = result["cleaned_tokens"]
    highlighted_html = result["highlighted_html"]
    status = result["status"]
    msg = result["message"]

    if status != "success":
        return (
            f"⚠️ **Notice:** {msg}",
            confidence,
            probs,
            highlighted_html,
            cleaned
        )

    emoji = "🟢" if sentiment.lower() == "positive" else "🔴"
    sentiment_md = f"### {emoji} **Predicted Sentiment:** {sentiment}\n**Confidence:** {confidence:.2f}%"

    return sentiment_md, confidence, probs, highlighted_html, cleaned


def process_batch_file(file_obj: Any) -> Tuple[pd.DataFrame, str, Any]:
    """Batch CSV/Excel file processing handler."""
    if INIT_ERROR or predictor is None:
        return pd.DataFrame(), "❌ Model not initialized.", None

    if file_obj is None:
        return pd.DataFrame(), "⚠️ Please upload a valid CSV or Excel file.", None

    try:
        df, counts, out_path = predictor.predict_batch(file_obj.name)
        total = len(df)
        pos_cnt = counts.get("Positive", 0)
        neg_cnt = counts.get("Negative", 0)

        summary_md = (
            f"### 📊 Batch Analysis Complete\n"
            f"- **Total Reviews Analyzed:** `{total:,}`\n"
            f"- **Positive Sentiment:** `{pos_cnt:,}` ({pos_cnt/total*100:.1f}%)\n"
            f"- **Negative Sentiment:** `{neg_cnt:,}` ({neg_cnt/total*100:.1f}%)\n"
        )
        return df, summary_md, out_path
    except Exception as exc:
        return pd.DataFrame(), f"❌ Batch Processing Failed: {exc}", None


def get_feature_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top positive and negative vocabulary features."""
    if predictor is None:
        return pd.DataFrame(), pd.DataFrame()
    return predictor.get_top_features(top_n=15)


# Custom CSS styling
custom_css = """
.container { max-width: 1000px; margin: auto; }
.title-box { text-align: center; margin-bottom: 1rem; }
"""

# Build Gradio Multi-Tab Blocks Interface
with gr.Blocks(title="AI Sentiment Analysis") as demo:
    gr.Markdown(
        """
        # 🎭 AI Sentiment Analysis & Explainable AI Dashboard
        ### Machine Learning Based Text Sentiment Classification & Batch Analytics
        """
    )

    with gr.Tabs():
        # TAB 1: Single Text Analysis & Explainable AI
        with gr.TabItem("🔍 Single Sentence Predictor & Explainable AI"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_box = gr.Textbox(
                        lines=4,
                        placeholder="Type your review or text here (e.g. 'I absolutely loved this movie. It was amazing!')...",
                        label="Input Text / Customer Review"
                    )
                    submit_btn = gr.Button("⚡ Analyze Sentiment", variant="primary")

                with gr.Column(scale=2):
                    sentiment_output = gr.Markdown(value="*Submit text to view prediction*", label="Prediction Result")
                    confidence_output = gr.Number(label="Confidence Score (%)", precision=2)
                    probabilities_output = gr.Label(label="Class Probability Distribution", num_top_classes=2)

            st_title = gr.Markdown("### 💡 Explainable AI (Keyword Highlighting)")
            st_sub = gr.Markdown("*Green pills indicate positive contributing words; Red pills indicate negative contributing words.*")
            highlight_output = gr.HTML(value="<em>Highlighted keywords will appear here...</em>")
            tokens_output = gr.Textbox(label="Cleaned & Lemmatized Tokens", interactive=False)

            submit_btn.click(
                fn=analyze_single_text,
                inputs=[input_box],
                outputs=[sentiment_output, confidence_output, probabilities_output, highlight_output, tokens_output]
            )

            gr.Examples(
                examples=[
                    ["I absolutely loved this movie. It was amazing!"],
                    ["Terrible experience. The service was slow and workers were rude."],
                    ["The plot was okay, but the acting was somewhat weak."],
                    ["Great quality product, fast shipping, and awesome support!"]
                ],
                inputs=[input_box],
                outputs=[sentiment_output, confidence_output, probabilities_output, highlight_output, tokens_output],
                fn=analyze_single_text,
                cache_examples=False
            )

        # TAB 2: Batch CSV File Processing & Export
        with gr.TabItem("📁 Batch CSV Analysis & Export"):
            gr.Markdown(
                """
                ### 📤 Upload Dataset File (.csv / .xlsx)
                Upload a CSV or Excel file containing customer reviews or tweets to analyze all rows at once.
                """
            )
            file_input = gr.File(label="Upload CSV or Excel File", file_types=[".csv", ".xlsx", ".xls"])
            batch_btn = gr.Button("🚀 Process Batch Dataset", variant="primary")
            batch_summary = gr.Markdown(value="*Upload a file and click Process to generate batch predictions.*")
            batch_df_preview = gr.DataFrame(label="Annotated Predictions Preview", wrap=True)
            batch_download = gr.File(label="📥 Download Annotated CSV Results")

            batch_btn.click(
                fn=process_batch_file,
                inputs=[file_input],
                outputs=[batch_df_preview, batch_summary, batch_download]
            )

        # TAB 3: Model Architecture & Feature Importance
        with gr.TabItem("📊 Model Architecture & Vocabulary Weights"):
            gr.Markdown("### 🏆 Top Model Vocabulary Coefficients (Feature Importance)")
            with gr.Row():
                pos_df, neg_df = get_feature_tables()
                with gr.Column():
                    gr.Markdown("#### 🟢 Top 15 Positive Keywords")
                    gr.DataFrame(value=pos_df, interactive=False)
                with gr.Column():
                    gr.Markdown("#### 🔴 Top 15 Negative Keywords")
                    gr.DataFrame(value=neg_df, interactive=False)

            gr.Markdown(
                """
                ---
                ### 🏗️ Machine Learning Pipeline Specifications
                - **Algorithm:** Logistic Regression (L2 Regularized) + Probability Calibration
                - **Vocabulary:** 150,000 TF-IDF Unigram & Bigram Features
                - **Training Datasets:** 88,000+ multi-domain samples (IMDB, SST-2, Penn Treebank Trees, TextAnalytics)
                - **Accuracy:** **90.44%** 5-Fold Cross-Validation Accuracy
                """
            )

if __name__ == "__main__":
    demo.launch(css=custom_css)
