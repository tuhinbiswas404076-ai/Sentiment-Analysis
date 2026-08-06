# AI Sentiment Analysis API

A professional Flask REST API for 3-class sentiment classification using a trained Logistic Regression model and TF-IDF vectorizer.

## Features
- Single-text prediction
- Batch prediction
- Confidence scores and class probabilities
- Input validation and structured error handling
- SQLite-based prediction history
- Analytics endpoint
- Swagger UI documentation
- Docker support
- Basic rate limiting and CORS protection

## Project structure
- app.py: Flask application and API routes
- utils/preprocessing.py: preprocessing logic compatible with the training notebook
- models/: trained model artifacts
- templates/: web UI
- static/: CSS and JavaScript

## Installation
```bash
pip install -r requirements.txt
```

## Run locally
```bash
python app.py
```

## Run with Gunicorn
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

## Docker
```bash
docker build -t sentiment-api .
docker run -p 5000:5000 sentiment-api
```

## API examples
### Health
```bash
curl http://localhost:5000/health
```

### Predict
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text":"I really enjoyed this movie!"}'
```

### Batch predict
```bash
curl -X POST http://localhost:5000/predict/batch -H "Content-Type: application/json" -d '{"texts":["I love this product","This is terrible","The product is okay"]}'
```
