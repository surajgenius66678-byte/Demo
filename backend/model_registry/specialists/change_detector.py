"""
OSCD bi-temporal change-detection specialist.

Supports two input formats:

1. Original OSCD dataset folders:
       imgs_1_rect/
           B01.tif
           B02.tif
           ...
           B8A.tif
           ...
           B12.tif

2. SatQuery production Tile.array_path:
       normalized 13-band GeoTIFF

The original OSCD folder format is used for training/evaluation.

The single GeoTIFF format is used by the Part 4 inference pipeline after
tile_image() has produced normalized multi-band GeoTIFF tiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from torchgeo.models import FCSiamDiff


class OSCDChangeDetector:
    """
    SatQuery specialist wrapper around TorchGeo FCSiamDiff.

    The model expects a bi-temporal pair:

        [2, 13, H, W]

    using the 13 OSCD/Sentinel-2 bands:

        B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12

    Supported inputs:

    - Original OSCD image folders containing the 13 individual band files.
    - Single 13-band GeoTIFF tiles produced by SatQuery preprocessing.
    """

    OSCD_BANDS = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    )

    EXPECTED_BANDS = 13

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.model = FCSiamDiff(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=self.EXPECTED_BANDS,
            classes=1,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # OSCD folder loader
    # ------------------------------------------------------------------

    def _load_oscd_folder(
        self,
        folder: str,
    ) -> torch.Tensor:
        """
        Load an original OSCD 13-band image folder.

        The same percentile normalization used during training is applied.
        """

        folder_path = Path(folder)

        if not folder_path.exists():
            raise FileNotFoundError(
                f"OSCD image folder not found: {folder_path}"
            )

        if not folder_path.is_dir():
            raise ValueError(
                f"Expected OSCD image folder, got: {folder_path}"
            )

        loaded: list[np.ndarray] = []

        for band_name in self.OSCD_BANDS:
            band_path = folder_path / f"{band_name}.tif"

            if not band_path.exists():
                raise FileNotFoundError(
                    f"Missing OSCD band: {band_path}"
                )

            with rasterio.open(band_path) as src:
                array = src.read(1).astype(np.float32)

            loaded.append(array)

        image = np.stack(
            loaded,
            axis=0,
        )

        # Same normalization used during training.
        for index in range(image.shape[0]):
            band = image[index]

            low = np.percentile(
                band,
                2,
            )

            high = np.percentile(
                band,
                98,
            )

            if high > low:
                image[index] = np.clip(
                    (band - low) / (high - low),
                    0.0,
                    1.0,
                )
            else:
                image[index] = 0.0

        return torch.from_numpy(image)

    # ------------------------------------------------------------------
    # SatQuery GeoTIFF loader
    # ------------------------------------------------------------------

    def _load_geotiff(
        self,
        file_path: str,
    ) -> torch.Tensor:
        """
        Load a normalized SatQuery multi-band GeoTIFF.

        tile_image() already performs normalization before writing the tile,
        therefore normalization is intentionally NOT repeated here.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Change-detection tile not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Expected GeoTIFF file, got: {path}"
            )

        with rasterio.open(path) as src:
            array = src.read().astype(np.float32)

        if array.ndim != 3:
            raise ValueError(
                "Expected raster with shape "
                "[bands, height, width], "
                f"got shape={array.shape} for {path}"
            )

        band_count = array.shape[0]

        if band_count != self.EXPECTED_BANDS:
            raise ValueError(
                "OSCDChangeDetector requires exactly "
                f"{self.EXPECTED_BANDS} bands, but {path} contains "
                f"{band_count} band(s)."
            )

        return torch.from_numpy(array)

    # ------------------------------------------------------------------
    # Unified image loader
    # ------------------------------------------------------------------

    def _load_image(
        self,
        path: str,
    ) -> torch.Tensor:
        """
        Load either an original OSCD folder or a SatQuery GeoTIFF.
        """

        target = Path(path)

        if target.is_dir():
            return self._load_oscd_folder(
                str(target)
            )

        if target.is_file():
            return self._load_geotiff(
                str(target)
            )

        raise FileNotFoundError(
            f"Change-detection input does not exist: {target}"
        )

    # ------------------------------------------------------------------
    # Change VQA / natural-language answer
    # ------------------------------------------------------------------

    def answer(
        self,
        prompt: str,
        change_results: list[dict],
    ) -> str:
        """
        Produce an evidence-grounded answer from change-detection results.

        The FCSiamDiff model provides pixel-level change probabilities,
        not semantic object labels. Therefore this method must not invent
        claims such as "a building was demolished" unless another specialist
        provides that evidence.
        """

        if not change_results:
            return (
                "No valid before-and-after tile results were available "
                "to answer the change question."
            )

        changed_area_px = sum(
            int(
                result.get(
                    "changed_area_px",
                    0,
                )
            )
            for result in change_results
        )

        percentages = [
            float(
                result.get(
                    "changed_area_pct",
                    0.0,
                )
            )
            for result in change_results
        ]

        if not percentages:
            changed_pct = 0.0
        else:
            changed_pct = sum(percentages) / len(percentages)

        prompt_lower = prompt.lower().strip()

        # --------------------------------------------------------------
        # Percentage / area questions
        # --------------------------------------------------------------

        if any(
            phrase in prompt_lower
            for phrase in (
                "how much",
                "how large",
                "what percentage",
                "percentage",
                "percent",
                "%",
                "area",
                "extent",
                "how much area",
            )
        ):
            return (
                f"Approximately {changed_pct:.2f}% of the compared "
                "area is classified as changed."
            )

        # --------------------------------------------------------------
        # Yes / no change questions
        # --------------------------------------------------------------

        if any(
            phrase in prompt_lower
            for phrase in (
                "did anything change",
                "is there a change",
                "is there any change",
                "any change",
                "whether anything changed",
                "has anything changed",
                "was there a change",
            )
        ):
            if changed_pct > 0.0:
                return (
                    "Yes. The model detected change in approximately "
                    f"{changed_pct:.2f}% of the compared area."
                )

            return (
                "No changed area was detected by the change-detection model."
            )

        # --------------------------------------------------------------
        # Generic change-description request
        # --------------------------------------------------------------

        return (
            f"The model detected approximately {changed_pct:.2f}% "
            "changed area between the two images. "
            "The available change-detection evidence does not identify "
            "the specific objects responsible for the detected change."
        )

    # ------------------------------------------------------------------
    # Change detection inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect_change(
        self,
        before_path: str,
        after_path: str,
    ) -> dict:
        """
        Run bi-temporal change detection.

        Returns a dictionary compatible with ChangeDetectionAdapter.
        """

        before = self._load_image(
            before_path
        )

        after = self._load_image(
            after_path
        )

        # --------------------------------------------------------------
        # Validate spatial compatibility
        # --------------------------------------------------------------

        if before.shape != after.shape:
            raise ValueError(
                "Before/after change-detection inputs must have identical "
                f"shapes, got before={tuple(before.shape)} and "
                f"after={tuple(after.shape)}"
            )

        if before.ndim != 3:
            raise ValueError(
                "Expected each image to have shape "
                "[13, height, width], got "
                f"{tuple(before.shape)}"
            )

        if before.shape[0] != self.EXPECTED_BANDS:
            raise ValueError(
                "Expected 13 input bands, got "
                f"{before.shape[0]}"
            )

        # --------------------------------------------------------------
        # Build FCSiamDiff input
        #
        # [13, H, W]
        #       ↓
        # [2, 13, H, W]
        #       ↓
        # [1, 2, 13, H, W]
        # --------------------------------------------------------------

        pair = torch.stack(
            [
                before,
                after,
            ],
            dim=0,
        ).unsqueeze(0)

        pair = pair.to(
            self.device,
            dtype=torch.float32,
        )

        # --------------------------------------------------------------
        # Model inference
        # --------------------------------------------------------------

        logits = self.model(
            pair
        )

        probability = torch.sigmoid(
            logits
        )

        mask = probability >= 0.5

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        changed_area_px = int(
            mask.sum().item()
        )

        total_pixels = int(
            mask.numel()
        )

        changed_area_pct = (
            changed_area_px / total_pixels * 100.0
            if total_pixels > 0
            else 0.0
        )

        mean_confidence = float(
            probability.mean().item()
        )

        # --------------------------------------------------------------
        # Adapter-compatible result
        # --------------------------------------------------------------

        return {
            "probability_raster_path": None,
            "changed_area_px": changed_area_px,
            "changed_area_pct": changed_area_pct,
            "mean_confidence": mean_confidence,
        }