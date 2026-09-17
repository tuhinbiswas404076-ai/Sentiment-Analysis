import sys
from pathlib import Path

import streamlit as st

# Setup sys.path to allow imports from deployment/utils or utils
BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = BASE_DIR / "deployment"
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from utils.preprocessing import load_model_artifacts, preprocess_text

# Page Configuration
st.set_page_config(
    page_title="Sentiment Analysis AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern visual design
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .positive-box {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid #10b981;
        color: #10b981;
    }
    .negative-box {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        color: #ef4444;
    }
    .sentiment-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_artifacts():
    """Cache loaded model artifacts for fast prediction response times."""
    return load_model_artifacts()


def main():
    st.markdown('<div class="main-header">🎭 Sentiment Analysis AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter any sentence, review, or tweet to analyze its sentiment in real-time.</div>', unsafe_allow_html=True)

    # Sidebar Information
    with st.sidebar:
        st.header("🤖 Model Architecture")
        st.markdown("""
        **Pipeline Specs:**
        - **Algorithm:** Logistic Regression (L2 Regularized)
        - **Vectorization:** TF-IDF (Unigram + Bigram, 150k features)
        - **Training Data:** 88,000+ samples (IMDB + SST-2 + PTB trees)
        - **Target Metric:** Binary Sentiment Classification
        """)
        st.divider()
        st.caption("🚀 Deployed via Streamlit Community Cloud")

    # Load Model Artifacts
    try:
        model, vectorizer, encoder = get_artifacts()
    except Exception as e:
        st.error(f"❌ Failed to load model artifacts: {e}")
        st.stop()

    # Pre-set Sample Prompts
    st.write("**Quick Sample Inputs:**")
    col_s1, col_s2, col_s3 = st.columns(3)

    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""

    if col_s1.button("😍 Exceptional Experience"):
        st.session_state.input_text = "The story was absolutely breathtaking, with outstanding performances and stunning cinematography!"

    if col_s2.button("😡 Disappointing Outcome"):
        st.session_state.input_text = "Extremely terrible service, long delays, and completely unhelpful customer support."

    if col_s3.button("🤔 Mixed Thoughts"):
        st.session_state.input_text = "The special effects were awesome, but the plot felt weak and predictable."

    # User Input Text Area
    user_input = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.input_text,
        height=130,
        placeholder="Type something here (e.g. 'This app is fast, accurate, and easy to use!')",
        key="text_input_area"
    )

    col_btn, _ = st.columns([1, 4])
    analyze_clicked = col_btn.button("⚡ Analyze Sentiment", type="primary", use_container_width=True)

    if user_input.strip():
        if analyze_clicked or user_input:
            with st.spinner("Processing NLP pipeline & evaluating sentiment..."):
                cleaned = preprocess_text(user_input)
                features = vectorizer.transform([cleaned])
                probs = model.predict_proba(features)[0]
                pred_idx = model.predict(features)[0]
                sentiment = str(encoder.inverse_transform([pred_idx])[0]).lower()
                confidence = float(max(probs)) * 100

                # Determine Probabilities per class
                class_probs = dict(zip(encoder.classes_, probs))

            st.divider()

            # Display Sentiment Prediction Card
            is_pos = sentiment == "positive"
            box_class = "positive-box" if is_pos else "negative-box"
            emoji = "🟢 Positive" if is_pos else "🔴 Negative"

            st.markdown(
                f"""
                <div class="sentiment-box {box_class}">
                    <div class="sentiment-title">{emoji}</div>
                    <div>Confidence Level: <strong>{confidence:.2f}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display Detailed Metrics & Probability Distribution
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Class Probabilities")
                for cls_name, prob in sorted(class_probs.items()):
                    st.write(f"**{cls_name.capitalize()}**")
                    st.progress(float(prob))
                    st.caption(f"Probability: {prob * 100:.2f}%")

            with col2:
                st.subheader("🔍 Preprocessing Inspector")
                st.markdown("**Raw Input:**")
                st.info(user_input)
                st.markdown("**Cleaned & Lemmatized Tokens:**")
                st.code(cleaned if cleaned else "(No valid tokens remaining after stopword removal)", language="text")


if __name__ == "__main__":
    main()
