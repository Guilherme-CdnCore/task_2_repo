from fastapi.testclient import TestClient
from SpaceX_ETL.Backend.api import app


client = TestClient(app)


def test_launches_endpoint_ok():
    resp = client.get("/launches?limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if data:
        item = data[0]
        assert "id" in item
        assert "date_lisbon" in item

