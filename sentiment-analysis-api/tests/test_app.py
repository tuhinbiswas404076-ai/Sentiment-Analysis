import json

import pytest

from app import app


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json()['success'] is True


def test_predict_endpoint(client):
    response = client.post('/predict', json={'text': 'I really loved this experience'})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['sentiment'] in {'positive', 'neutral', 'negative'}


def test_batch_predict_endpoint(client):
    response = client.post('/predict/batch', json={'texts': ['Great product', 'Bad service', 'It is okay']})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['summary']['total_texts'] == 3
