from pathlib import Path

from fastapi.testclient import TestClient

from entailforge import api


def test_api_health_and_prediction(tiny_training_run, monkeypatch):
    _, report = tiny_training_run
    checkpoint = Path(report["checkpoint"])
    monkeypatch.setenv("ENTAILFORGE_CHECKPOINT", str(checkpoint))
    api._get_predictor.cache_clear()
    client = TestClient(api.app)

    health = client.get("/health")
    response = client.post(
        "/predict",
        json={
            "premises": [
                "All poets are readers.",
                "No readers are silent.",
            ],
            "hypothesis": "No poets are silent.",
            "explain": False,
        },
    )

    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert response.status_code == 200
    assert response.json()["label"] in {"entailed", "not_entailed"}
