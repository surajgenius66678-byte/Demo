from fastapi.testclient import TestClient

from backend.api.main import app, image_store
from backend.shared.schemas import (
    ImageMetadata,
    Modality,
)


def test_query_endpoint_creates_and_completes_job():
    image = ImageMetadata(
        image_id="api-test-image",
        modality=Modality.OPTICAL,
        crs="EPSG:4326",
        bounds=[0.0, 0.0, 1.0, 1.0],
        width=512,
        height=512,
        band_count=3,
        dtype="uint8",
        resolution_m=1.0,
        timestamp=None,
        cog_path="/fake/api-test.tif",
        is_valid=True,
        validation_errors=[],
    )

    image_store.put(image)

    client = TestClient(app)

    response = client.post(
        "/api/query",
        json={
            "query": "What is visible in this image?",
            "image_ids": ["api-test-image"],
        },
    )

    assert response.status_code == 200

    job_id = response.json()["job_id"]
    assert job_id

    job_response = client.get(f"/api/jobs/{job_id}")

    assert job_response.status_code == 200

    data = job_response.json()
    assert data["status"] in {"queued", "running", "done"}

    trace_response = client.get(f"/api/jobs/{job_id}/trace")

    assert trace_response.status_code == 200
    assert "steps" in trace_response.json()
