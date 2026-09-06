from __future__ import annotations

import pytest

from backend.shared.schemas import Modality, ModelRegistryEntry, TaskType, Tile


def make_tile(tile_id: str, image_id: str, col_off: int = 0, row_off: int = 0, width: int = 512, height: int = 512) -> Tile:
    return Tile(
        tile_id=tile_id,
        image_id=image_id,
        col_off=col_off,
        row_off=row_off,
        width=width,
        height=height,
        affine_transform=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        array_path=f"/fake/tiles/{tile_id}.npy",
    )


@pytest.fixture
def single_tile() -> Tile:
    return make_tile("t-0000", "img-optical-1")


@pytest.fixture
def four_tiles() -> list[Tile]:
    return [make_tile(f"t-000{i}", "img-optical-1", col_off=i * 512) for i in range(4)]


@pytest.fixture
def before_after_tiles() -> list[Tile]:
    """Two co-located tiles from two different (bi-temporal) images."""
    return [
        make_tile("t-before-0", "img-2024", col_off=0, row_off=0),
        make_tile("t-after-0", "img-2025", col_off=0, row_off=0),
    ]


@pytest.fixture
def optical_sar_tiles() -> list[Tile]:
    """Two co-located tiles, one optical one SAR."""
    return [
        make_tile("t-opt-0", "img-optical-1", col_off=0, row_off=0),
        make_tile("t-sar-0", "img-sar-1", col_off=0, row_off=0),
    ]


@pytest.fixture
def sample_registry() -> list[ModelRegistryEntry]:
    return [
        ModelRegistryEntry(
            name="test-vlm", version="0.0.1",
            tasks=[TaskType.SINGLE_IMAGE_VQA, TaskType.CAPTIONING],
            modalities=[Modality.OPTICAL, Modality.SAR],
            checkpoint_path="(test)", quantization="none",
            requires_coregistration=False, max_input_px=1024,
        ),
        ModelRegistryEntry(
            name="test-grounding", version="0.0.1",
            tasks=[TaskType.GROUNDING],
            modalities=[Modality.OPTICAL, Modality.SAR],
            checkpoint_path="(test)", quantization="none",
            requires_coregistration=False, max_input_px=1024,
        ),
    ]
