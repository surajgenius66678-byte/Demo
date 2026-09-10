"""
Part 4 — model loader.

Owns lazy GPU loading, LRU eviction under a VRAM budget, and quantized
loading. Model construction is delegated to a model factory so SatQuery is
not coupled to one checkpoint or parameter count.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from backend.model_registry.exceptions import ModelLoadError
from backend.shared.schemas import ModelRegistryEntry

logger = logging.getLogger("satquery.model_registry.loader")

# (entry, device, quantization_override) -> loaded model handle.
ModelFactory = Callable[[ModelRegistryEntry, str, Optional[str]], Any]


@dataclass
class LoadedModel:
    entry: ModelRegistryEntry
    handle: Any
    quantization: str
    vram_mb: float
    last_used: float = field(default_factory=time.monotonic)


class _HuggingFaceVLMHandle:
    """Small, model-family-neutral HF VLM handle used by VLMAdapter.

    The rest of SatQuery only sees ``generate(image_path, prompt)``. The
    checkpoint can therefore be changed without changing Part 2, the API,
    evidence, or the VLM adapter. Architecture-specific logic stays inside
    this handle and is selected from the checkpoint config when possible.
    """

    def __init__(self, model: Any, processor: Any, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.vram_mb = 0.0

    def generate(self, image_path: str, prompt: str) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ModelLoadError("Pillow is required for Hugging Face VLM inference.") from exc

        image = Image.open(Path(image_path)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Modern Transformers processors for multimodal causal LMs expose
        # apply_chat_template. Keep all checkpoint-specific preprocessing
        # behind this interface.
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            # Older Qwen2-VL releases use qwen-vl-utils for vision packing.
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as exc:
                raise ModelLoadError(
                    "This VLM processor needs qwen-vl-utils. Install it or use a "
                    "checkpoint compatible with the installed Transformers version."
                ) from exc
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)

        generated = self.model.generate(**inputs, max_new_tokens=128)
        input_ids = getattr(inputs, "input_ids", None)
        if input_ids is not None:
            generated = generated[:, input_ids.shape[-1] :]
        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return {"text": text}

    def close(self) -> None:
        del self.model
        del self.processor
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _load_huggingface_vlm(entry: ModelRegistryEntry, device: str, quantization_override: str | None) -> Any:
    """Load any compatible HF image-text-to-text checkpoint.

    No model ID is hard-coded here. ``entry.checkpoint_path`` is the seam for
    the final 2B/3B/7B model or the friend's fine-tuned checkpoint.
    """
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ModelLoadError(
            "Hugging Face VLM loading requires torch and transformers."
        ) from exc

    checkpoint = os.getenv("SATQUERY_VLM_CHECKPOINT") or entry.checkpoint_path
    if not checkpoint or checkpoint.startswith("checkpoints/"):
        raise ModelLoadError(
            f"No real VLM checkpoint configured for '{entry.name}'. "
            "Set checkpoint_path to a Hugging Face/local checkpoint or "
            "SATQUERY_VLM_CHECKPOINT."
        )

    try:
        # AutoModelForImageTextToText is the preferred generic entry point in
        # current Transformers. Fall back to Qwen2VL explicitly for older
        # Transformers releases because Qwen2-VL is a common SatQuery VLM.
        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration
            model_cls = Qwen2VLForConditionalGeneration

        quant = quantization_override or entry.quantization
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if quant == "4bit":
            kwargs["load_in_4bit"] = True
            kwargs["torch_dtype"] = torch.bfloat16
        elif quant == "8bit":
            kwargs["load_in_8bit"] = True
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model = model_cls.from_pretrained(checkpoint, **kwargs)
        processor = AutoProcessor.from_pretrained(checkpoint)
        return _HuggingFaceVLMHandle(model, processor, device)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load VLM checkpoint '{checkpoint}': {exc}"
        ) from exc


def default_model_factory(entry: ModelRegistryEntry, device: str, quantization_override: str | None) -> Any:
    """Generic factory; model identity comes from registry/config, not code."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ModelLoadError(
            f"Cannot load '{entry.name}': torch is not installed."
        ) from exc

    if any(task.value in {"SINGLE_IMAGE_VQA", "CAPTIONING"} for task in entry.tasks):
        return _load_huggingface_vlm(entry, device, quantization_override)

    raise ModelLoadError(
        f"No generic loader is registered for model '{entry.name}'. "
        "Add a task-specific factory for this specialist without changing "
        "the model registry or agent contracts."
    )


class ModelLoader:
    """Lazy-load models with LRU eviction under a VRAM budget."""

    def __init__(
        self,
        registry: list[ModelRegistryEntry],
        vram_budget_mb: float = 8192.0,
        model_factory: ModelFactory = default_model_factory,
        device: str = "cuda",
    ):
        self.registry: dict[str, ModelRegistryEntry] = {e.name: e for e in registry}
        self.vram_budget_mb = vram_budget_mb
        self._model_factory = model_factory
        self.device = device
        self._resident: dict[str, LoadedModel] = {}

    def get(self, model_name: str, quantization_override: str | None = None) -> Any:
        cached = self._resident.get(model_name)
        if cached is not None:
            if quantization_override is None or cached.quantization == quantization_override:
                cached.last_used = time.monotonic()
                return cached.handle
            self.unload(model_name)

        if model_name not in self.registry:
            raise ModelLoadError(f"'{model_name}' is not in the registry.")
        entry = self.registry[model_name]
        return self._load(entry, quantization_override).handle

    def _load(self, entry: ModelRegistryEntry, quantization_override: str | None) -> LoadedModel:
        handle = self._model_factory(entry, self.device, quantization_override)
        vram_mb = float(getattr(handle, "vram_mb", 0.0)) or _estimate_vram_mb(entry, quantization_override)
        self._evict_lru_until_fits(vram_mb)
        if not self._resident and self.vram_used_mb() + vram_mb > self.vram_budget_mb:
            logger.warning(
                "%s alone (%.0f MB) exceeds the %.0f MB budget; loading anyway (best effort).",
                entry.name, vram_mb, self.vram_budget_mb,
            )

        loaded = LoadedModel(
            entry=entry,
            handle=handle,
            quantization=quantization_override or entry.quantization,
            vram_mb=vram_mb,
        )
        self._resident[entry.name] = loaded
        logger.info(
            "Loaded %s (%s, %.0f MB); resident=%s (%.0f/%.0f MB)",
            entry.name, loaded.quantization, vram_mb,
            list(self._resident), self.vram_used_mb(), self.vram_budget_mb,
        )
        return loaded

    def _evict_lru_until_fits(self, incoming_vram_mb: float) -> None:
        while self._resident and self.vram_used_mb() + incoming_vram_mb > self.vram_budget_mb:
            victim_name = min(self._resident, key=lambda n: self._resident[n].last_used)
            self.unload(victim_name)

    def unload(self, model_name: str) -> None:
        loaded = self._resident.pop(model_name, None)
        if loaded is not None:
            close = getattr(loaded.handle, "close", None)
            if callable(close):
                close()
            logger.info("Evicted %s, freed %.0f MB", model_name, loaded.vram_mb)

    def resident_models(self) -> list[str]:
        return list(self._resident.keys())

    def vram_used_mb(self) -> float:
        return sum(m.vram_mb for m in self._resident.values())

    def health(self) -> dict:
        return {
            "models_loaded": self.resident_models(),
            "vram_used_mb": self.vram_used_mb(),
            "vram_budget_mb": self.vram_budget_mb,
        }


def _estimate_vram_mb(entry: ModelRegistryEntry, quantization_override: str | None) -> float:
    quant = quantization_override or entry.quantization
    base_mb = {"none": 6000.0, "8bit": 3200.0, "4bit": 1800.0}
    return base_mb.get(quant, 4000.0)
