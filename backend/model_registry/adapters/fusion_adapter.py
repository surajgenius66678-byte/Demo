"""
Fusion adapter — optical + SAR fusion.

Pairs optical and SAR tiles using their shared spatial offsets, runs the
registered fusion specialist on each matched pair, and converts the raw
model output into the shared Evidence schema.

The current fusion specialist is TorchGeo CROMA. CROMA is a pretrained
optical-SAR representation model: it produces optical, SAR, and joint
embeddings, but it does not itself produce detections or natural-language
answers.

Therefore this adapter does NOT fabricate detections or text from
embeddings.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.model_registry.adapters.base import BaseAdapter, confidence_band
from backend.shared.schemas import (
    Confidence,
    Evidence,
    Modality,
    ModelRegistryEntry,
    TaskType,
    Tile,
)


class FusionAdapter(BaseAdapter):
    """Adapter for optical + SAR cross-modal fusion."""

    handles = (TaskType.OPTICAL_SAR_FUSION,)

    def __init__(
        self,
        query: str | None = None,
        image_modalities: dict[str, Modality] | None = None,
    ) -> None:
        self.query = query
        self.image_modalities = image_modalities or {}

    def preprocess(
        self,
        tiles: list[Tile],
        entry: ModelRegistryEntry,
    ) -> Any:
        """
        Group tiles by image, identify the optical and SAR images,
        and pair tiles that share the same spatial offset.
        """
        by_image: dict[str, list[Tile]] = defaultdict(list)

        for tile in tiles:
            by_image[tile.image_id].append(tile)

        if len(by_image) != 2:
            raise ValueError(
                "Fusion needs tiles from exactly 2 images "
                f"(optical + SAR), got {len(by_image)}."
            )

        optical_id, sar_id = self._split_by_modality(
            list(by_image)
        )

        pairs = self._pair_by_offset(
            by_image[optical_id],
            by_image[sar_id],
        )

        return {
            "pairs": pairs,
            "prompt": self.query,
        }

    def _split_by_modality(
        self,
        image_ids: list[str],
    ) -> tuple[str, str]:
        """
        Resolve which input image is OPTICAL and which is SAR.

        Fusion requires exactly one image of each modality.
        We intentionally do not guess based on image ID ordering.
        """
        labeled = {
            image_id: self.image_modalities.get(image_id)
            for image_id in image_ids
        }

        optical = [
            image_id
            for image_id, modality in labeled.items()
            if modality == Modality.OPTICAL
        ]

        sar = [
            image_id
            for image_id, modality in labeled.items()
            if modality == Modality.SAR
        ]

        if len(optical) == 1 and len(sar) == 1:
            return optical[0], sar[0]

        raise ValueError(
            "Fusion requires exactly one OPTICAL image and one SAR image. "
            f"Received modalities: {labeled}"
        )

    @staticmethod
    def _pair_by_offset(
        optical_tiles: list[Tile],
        sar_tiles: list[Tile],
    ) -> list[tuple[Tile, Tile]]:
        """
        Pair optical and SAR tiles using their shared spatial offsets.

        A pair is considered spatially corresponding when both tiles have
        the same (col_off, row_off).
        """
        sar_by_offset = {
            (tile.col_off, tile.row_off): tile
            for tile in sar_tiles
        }

        return [
            (
                optical_tile,
                sar_by_offset[
                    (
                        optical_tile.col_off,
                        optical_tile.row_off,
                    )
                ],
            )
            for optical_tile in optical_tiles
            if (
                optical_tile.col_off,
                optical_tile.row_off,
            ) in sar_by_offset
        ]

    def infer(
        self,
        model: Any,
        model_input: Any,
    ) -> Any:
        """
        Run the registered fusion model for every matched tile pair.

        The fusion model must expose:

            model.fuse(
                optical_path=str,
                sar_path=str,
                prompt=str | None,
            )
        """
        results = []

        for optical_tile, sar_tile in model_input["pairs"]:
            raw = model.fuse(
                optical_path=optical_tile.array_path,
                sar_path=sar_tile.array_path,
                prompt=model_input["prompt"],
            )

            results.append(
                {
                    "optical": optical_tile,
                    "sar": sar_tile,
                    "raw": raw,
                }
            )

        return results

    def postprocess(
        self,
        raw_output: Any,
        tiles: list[Tile],
        entry: ModelRegistryEntry,
        modality_used: list[Modality],
    ) -> Evidence:
        """
        Convert CROMA fusion representations into auditable Evidence.

        CROMA is an encoder rather than a detector or language model.
        It produces optical, SAR, and joint representations.

        We therefore report successful joint representations as evidence
        that optical-SAR fusion was successfully executed.

        We do NOT fabricate:
        - object detections
        - bounding boxes
        - masks
        - natural-language answers
        """
        joint_count = 0
        optical_count = 0
        sar_count = 0

        for item in raw_output:
            raw = item.get("raw", {})

            if raw.get("joint_GAP") is not None:
                joint_count += 1

            if raw.get("optical_GAP") is not None:
                optical_count += 1

            if raw.get("sar_GAP") is not None:
                sar_count += 1

        pair_count = len(raw_output)

        representation_ready = joint_count > 0

        warnings: list[str] = []

        if pair_count == 0:
            warnings.append(
                "No spatially matching optical-SAR tile pairs were available."
            )

        if not representation_ready:
            warnings.append(
                "CROMA did not return a joint optical-SAR representation."
            )

        # This represents successful execution of the fusion encoder.
        # It is NOT a semantic prediction probability.
        confidence_value = 1.0 if representation_ready else 0.0

        return Evidence(
            task=TaskType.OPTICAL_SAR_FUSION,
            model_used=entry.name,
            modality_used=modality_used,
            detections=[],
            vqa_answer_raw=None,
            stats={
                "tile_pairs_fused": float(pair_count),
                "joint_representations": float(joint_count),
                "optical_representations": float(optical_count),
                "sar_representations": float(sar_count),
            },
            confidence=Confidence(
                value=confidence_value,
                band=confidence_band(confidence_value),
                basis=(
                    "successful pretrained CROMA joint optical-SAR "
                    "representation; not a semantic prediction probability"
                ),
            ),
            warnings=warnings,
        )