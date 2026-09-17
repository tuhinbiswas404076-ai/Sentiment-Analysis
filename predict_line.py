import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
API_DIR = BASE_DIR / "sentiment-analysis-api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from utils.preprocessing import load_model_artifacts, preprocess_text

def main():
    print("=" * 60)
    print("       INTERACTIVE SENTIMENT PREDICTOR")
    print("=" * 60)
    print("Loading model artifacts...")
    model, vectorizer, encoder = load_model_artifacts()
    print("Model loaded successfully!\n")

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        cleaned = preprocess_text(text)
        features = vectorizer.transform([cleaned])
        probs = model.predict_proba(features)[0]
        pred_idx = model.predict(features)[0]
        sentiment = encoder.inverse_transform([pred_idx])[0]
        confidence = max(probs) * 100

        print(f"Input Line : '{text}'")
        print(f"Prediction : {sentiment.upper()}")
        print(f"Confidence : {confidence:.2f}%")
        print("=" * 60)
        return

    print("Type any text or sentence to analyze its sentiment (or type 'exit' to quit):\n")
    while True:
        try:
            text = input("Enter line: ").strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit", "q"):
                print("Exiting Sentiment Predictor.")
                break

            cleaned = preprocess_text(text)
            features = vectorizer.transform([cleaned])
            probs = model.predict_proba(features)[0]
            pred_idx = model.predict(features)[0]
            sentiment = str(encoder.inverse_transform([pred_idx])[0])
            confidence = float(max(probs)) * 100

            print("-" * 50)
            print(f"PREDICTION : {sentiment.upper()}")
            print(f"CONFIDENCE : {confidence:.2f}%")
            print("-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
