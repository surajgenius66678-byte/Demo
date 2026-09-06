"""
VLM adapter — Section 3.4: single-image VQA + captioning.

Wraps a HF-style vision-language model (the specific base checkpoint is
one of Part 6's open decisions, Section 3.6 — this adapter only knows the
*shape* of that model's input/output, not which checkpoint is loaded).

Expected model interface (whatever `loader.get()` returns must provide):
    model.generate(image_path: str, prompt: str) -> {"text": str, "score": float}
"""
from __future__ import annotations

from typing import Any

from backend.model_registry.adapters.base import BaseAdapter, confidence_band
from backend.shared.schemas import Confidence, Evidence, Modality, ModelRegistryEntry, TaskType, Tile

_DEFAULT_CAPTION_PROMPT = "Describe this satellite image in one factual sentence."


class VLMAdapter(BaseAdapter):
    handles = (TaskType.SINGLE_IMAGE_VQA, TaskType.CAPTIONING)

    def __init__(self, query: str | None = None):
        # query set -> SINGLE_IMAGE_VQA; query unset -> CAPTIONING.
        self.query = query

    def preprocess(self, tiles: list[Tile], entry: ModelRegistryEntry) -> Any:
        if not tiles:
            raise ValueError("VLMAdapter.preprocess requires at least one tile.")
        return {"image_path": tiles[0].array_path, "prompt": self.query or _DEFAULT_CAPTION_PROMPT}

    def infer(self, model: Any, model_input: Any) -> Any:
        return model.generate(image_path=model_input["image_path"], prompt=model_input["prompt"])

    def postprocess(
        self, raw_output: Any, tiles: list[Tile], entry: ModelRegistryEntry, modality_used: list[Modality],
    ) -> Evidence:
        task = TaskType.SINGLE_IMAGE_VQA if self.query else TaskType.CAPTIONING
        text = raw_output.get("text", "") if isinstance(raw_output, dict) else str(raw_output)
        score = float(raw_output.get("score", 0.0)) if isinstance(raw_output, dict) else 0.0
        return Evidence(
            task=task,
            model_used=entry.name,
            modality_used=modality_used,
            vqa_answer_raw=text,
            stats={"generation_score": score} if score else {},
            confidence=Confidence(
                value=score or None,
                band=confidence_band(score),
                basis="VLM decoder confidence score",
            ),
            warnings=[] if text else ["Model returned an empty response."],
        )
