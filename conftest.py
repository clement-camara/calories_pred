import pytest
from app import app as flask_app


@pytest.fixture
def app():
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index(app, client):
    res = client.get('/')
    assert res.status_code == 200
    assert b"Diet App Bienvenue" in res.data


def test_predict(app, client):
    result = client.get('/predict/')
    assert result.status_code == 200
    assert result is not None
    assert result.request.method == 'GET'
