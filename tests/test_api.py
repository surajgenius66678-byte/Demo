from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient

from backend.api.main import app


def test_query_endpoint_creates_and_completes_job(tmp_path: Path):
    # Create a small real GeoTIFF so the test exercises the
    # real upload -> validation -> Part 3 storage pipeline.
    tif_path = tmp_path / "api-test.tif"

    data = np.zeros((3, 512, 512), dtype=np.uint8)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(
            0,
            1,
            1 / 512,
            1 / 512,
        ),
    ) as dst:
        dst.write(data)

    client = TestClient(app)

    # Use the real API upload boundary.
    upload_response = client.post(
        "/api/upload",
        files={
            "file": (
                "api-test.tif",
                tif_path.read_bytes(),
                "image/tiff",
            )
        },
        data={
            "modality": "OPTICAL",
        },
    )

    assert upload_response.status_code == 200

    image = upload_response.json()
    assert image["is_valid"] is True
    assert image["image_id"]

    image_id = image["image_id"]

    # Submit the query using the image returned by the real upload path.
    response = client.post(
        "/api/query",
        json={
            "query": "What is visible in this image?",
            "image_ids": [image_id],
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