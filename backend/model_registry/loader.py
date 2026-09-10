"""
Part 4 — model loader.

Owns lazy GPU loading, LRU eviction under a VRAM budget, and quantized
loading. Model construction is delegated to model factories so SatQuery
is not coupled to one checkpoint, model family, or parameter count.

Architecture:

    ModelRegistryEntry
            |
            v
    ModelLoader
            |
            v
    default_model_factory()
            |
       +----+-------------------+
       |                        |
       v                        v
  Specialist factory      Generic HF VLM
       |                        |
       +------------+-----------+
                    |
                    v
              Model handle
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from backend.model_registry.exceptions import ModelLoadError
from backend.shared.schemas import ModelRegistryEntry, TaskType

logger = logging.getLogger("satquery.model_registry.loader")


# ---------------------------------------------------------------------------
# Model factory contract
# ---------------------------------------------------------------------------

# (registry entry, device, quantization override) -> model handle
ModelFactory = Callable[
    [ModelRegistryEntry, str, Optional[str]],
    Any,
]


# Explicit specialist factories registered by model name.
#
# Example later:
#
# register_model_factory(
#     "satquery-grounding",
#     load_grounding_model,
# )
#
# This keeps specialist/model-family logic outside ModelLoader.
_SPECIALIST_FACTORIES: dict[str, ModelFactory] = {}


def register_model_factory(
    model_name: str,
    factory: ModelFactory,
) -> None:
    """
    Register a specialist model factory by registry model name.

    The factory receives:
        - ModelRegistryEntry
        - target device
        - optional quantization override

    and must return a model handle implementing the interface expected
    by the corresponding Part 4 adapter.
    """
    if not model_name:
        raise ValueError("model_name must be non-empty.")

    _SPECIALIST_FACTORIES[model_name] = factory

    logger.info(
        "Registered specialist model factory for '%s'.",
        model_name,
    )


def unregister_model_factory(model_name: str) -> None:
    """Remove a previously registered specialist factory."""
    _SPECIALIST_FACTORIES.pop(model_name, None)


def registered_model_factories() -> list[str]:
    """Return registered specialist model names."""
    return list(_SPECIALIST_FACTORIES.keys())


# ---------------------------------------------------------------------------
# Loaded model state
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """Runtime state for one model currently resident in memory."""

    entry: ModelRegistryEntry
    handle: Any
    quantization: str
    vram_mb: float
    last_used: int = 0


# ---------------------------------------------------------------------------
# Generic Hugging Face VLM
# ---------------------------------------------------------------------------


class _HuggingFaceVLMHandle:
    """
    Small, model-family-neutral Hugging Face VLM handle.

    VLMAdapter only sees:

        generate(image_path, prompt)

    Therefore the rest of SatQuery does not need to know which checkpoint
    or VLM family is being used.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: str,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.vram_mb = 0.0

    def generate(
        self,
        image_path: str,
        prompt: str,
    ) -> dict[str, Any]:
        """
        Run multimodal generation for one image.

        Returns:
            {
                "text": "...",
            }
        """
        try:
            from PIL import Image
        except ImportError as exc:
            raise ModelLoadError(
                "Pillow is required for Hugging Face VLM inference."
            ) from exc

        image = Image.open(Path(image_path)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        # Modern Transformers multimodal processors expose
        # apply_chat_template(..., tokenize=True).
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        except TypeError:
            # Older Qwen2-VL-style processors may require qwen-vl-utils.
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as exc:
                raise ModelLoadError(
                    "This VLM processor needs qwen-vl-utils. Install it or "
                    "use a checkpoint compatible with the installed "
                    "Transformers version."
                ) from exc

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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
            try:
                model_device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                model_device = self.device
            inputs = inputs.to(self.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=128,
        )

        input_ids = getattr(
            inputs,
            "input_ids",
            None,
        )

        if input_ids is not None:
            generated = generated[:, input_ids.shape[-1] :]

        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return {
            "text": text,
        }

    def close(self) -> None:
        """Release model resources."""
        try:
            del self.model
        except AttributeError:
            pass

        try:
            del self.processor
        except AttributeError:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except ImportError:
            pass


def _load_huggingface_vlm(
    entry: ModelRegistryEntry,
    device: str,
    quantization_override: str | None,
) -> Any:
    """
    Load a compatible Hugging Face image-text-to-text checkpoint.

    No checkpoint ID is hard-coded here.

    The checkpoint comes from:

        SATQUERY_VLM_CHECKPOINT
                    OR
        entry.checkpoint_path
    """
    try:
        import torch
        from transformers import AutoProcessor

    except ImportError as exc:
        raise ModelLoadError(
            "Hugging Face VLM loading requires torch and transformers."
        ) from exc

    checkpoint = (
        os.getenv("SATQUERY_VLM_CHECKPOINT")
        or entry.checkpoint_path
    )

    if not checkpoint or checkpoint.startswith("checkpoints/"):
        raise ModelLoadError(
            f"No real VLM checkpoint configured for '{entry.name}'. "
            "Set checkpoint_path to a Hugging Face/local checkpoint or "
            "SATQUERY_VLM_CHECKPOINT."
        )

    try:
        # Prefer the generic modern Transformers loader.
        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText

        except ImportError:
            # Compatibility path for older Transformers versions.
            from transformers import Qwen2VLForConditionalGeneration

            model_cls = Qwen2VLForConditionalGeneration

        quantization = (
            quantization_override
            or entry.quantization
        )

        kwargs: dict[str, Any] = {
            "device_map": "auto",
        }

        if quantization == "4bit":
            kwargs["load_in_4bit"] = True
            kwargs["torch_dtype"] = torch.bfloat16

        elif quantization == "8bit":
            kwargs["load_in_8bit"] = True
            kwargs["torch_dtype"] = torch.bfloat16

        else:
            kwargs["torch_dtype"] = (
                torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32
            )

        model = model_cls.from_pretrained(
            checkpoint,
            **kwargs,
        )

        processor = AutoProcessor.from_pretrained(
            checkpoint,
        )

        return _HuggingFaceVLMHandle(
            model=model,
            processor=processor,
            device=device,
        )

    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load VLM checkpoint '{checkpoint}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Default model factory
# ---------------------------------------------------------------------------


def default_model_factory(
    entry: ModelRegistryEntry,
    device: str,
    quantization_override: str | None,
) -> Any:
    """
    Resolve a registry entry to its appropriate model factory.

    Resolution order:

    1. Explicit specialist factory registered by model name.
    2. Generic Hugging Face VLM for VQA/captioning.
    3. Fail explicitly for unsupported specialist models.

    This prevents the loader from containing hard-coded model-specific
    implementation logic.
    """

    # ---------------------------------------------------------------
    # 1. Explicit specialist factory
    # ---------------------------------------------------------------

    specialist_factory = _SPECIALIST_FACTORIES.get(entry.name)

    if specialist_factory is not None:
        logger.info(
            "Using specialist factory for model '%s'.",
            entry.name,
        )

        return specialist_factory(
            entry,
            device,
            quantization_override,
        )

    # ---------------------------------------------------------------
    # 2. Generic HF VLM
    # ---------------------------------------------------------------

    if any(
        task in {
            TaskType.SINGLE_IMAGE_VQA,
            TaskType.CAPTIONING,
        }
        for task in entry.tasks
    ):
        return _load_huggingface_vlm(
            entry,
            device,
            quantization_override,
        )

    # ---------------------------------------------------------------
    # 3. No factory available
    # ---------------------------------------------------------------

    raise ModelLoadError(
        f"No model factory is registered for specialist "
        f"'{entry.name}'. "
        "Register one with register_model_factory()."
    )


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


class ModelLoader:
    """
    Lazy-load models with LRU eviction under a VRAM budget.

    The loader itself does not know how a specialist model works.
    It only manages:

        registry
        loading
        caching
        VRAM accounting
        LRU eviction
        unloading
    """

    def __init__(
        self,
        registry: list[ModelRegistryEntry],
        vram_budget_mb: float = 8192.0,
        model_factory: ModelFactory = default_model_factory,
        device: str |None=None,
    ):
        self.registry: dict[str, ModelRegistryEntry] = {
            entry.name: entry
            for entry in registry
        }

        self.vram_budget_mb = vram_budget_mb
        self._model_factory = model_factory
        if device is not None:
            self.device = device
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except (ImportError , OSError):
                self.device = "cpu"
        self._resident: dict[str, LoadedModel] = {}
        self._access_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        model_name: str,
        quantization_override: str | None = None,
    ) -> Any:
        """
        Return a loaded model handle.

        Models are loaded lazily on first access.
        """
        cached = self._resident.get(model_name)

        if cached is not None:
            if (
                quantization_override is None
                or cached.quantization == quantization_override
            ):
                self._access_counter += 1
                cached.last_used = self._access_counter
                return cached.handle

            # Quantization changed; reload.
            self.unload(model_name)

        if model_name not in self.registry:
            raise ModelLoadError(
                f"'{model_name}' is not in the registry."
            )

        entry = self.registry[model_name]

        return self._load(
            entry,
            quantization_override,
        ).handle

    def unload(
        self,
        model_name: str,
    ) -> None:
        """Unload one resident model."""
        loaded = self._resident.pop(
            model_name,
            None,
        )

        if loaded is None:
            return

        close = getattr(
            loaded.handle,
            "close",
            None,
        )

        if callable(close):
            close()

        logger.info(
            "Evicted %s, freed %.0f MB",
            model_name,
            loaded.vram_mb,
        )

    def resident_models(self) -> list[str]:
        """Return currently loaded model names."""
        return list(self._resident.keys())

    def vram_used_mb(self) -> float:
        """Return estimated VRAM currently used."""
        return sum(
            model.vram_mb
            for model in self._resident.values()
        )

    def health(self) -> dict[str, Any]:
        """Return model-loader health information."""
        return {
            "models_loaded": self.resident_models(),
            "vram_used_mb": self.vram_used_mb(),
            "vram_budget_mb": self.vram_budget_mb,
        }

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(
        self,
        entry: ModelRegistryEntry,
        quantization_override: str | None,
    ) -> LoadedModel:
        """Construct and register a model."""
        handle = self._model_factory(
            entry,
            self.device,
            quantization_override,
        )

        vram_mb = (
            float(
                getattr(
                    handle,
                    "vram_mb",
                    0.0,
                )
            )
            or _estimate_vram_mb(
                entry,
                quantization_override,
            )
        )

        self._evict_lru_until_fits(
            vram_mb,
        )

        # If the model itself exceeds the entire budget, load it anyway.
        if (
            not self._resident
            and self.vram_used_mb() + vram_mb
            > self.vram_budget_mb
        ):
            logger.warning(
                "%s alone (%.0f MB) exceeds the "
                "%.0f MB budget; loading anyway "
                "(best effort).",
                entry.name,
                vram_mb,
                self.vram_budget_mb,
            )
        self._access_counter += 1
        loaded = LoadedModel(
            entry=entry,
            handle=handle,
            quantization=(
                quantization_override
                or entry.quantization
            ),
            vram_mb=vram_mb,
            last_used=self._access_counter,
        )

        self._resident[entry.name] = loaded

        logger.info(
            "Loaded %s (%s, %.0f MB); resident=%s "
            "(%.0f/%.0f MB)",
            entry.name,
            loaded.quantization,
            vram_mb,
            list(self._resident),
            self.vram_used_mb(),
            self.vram_budget_mb,
        )

        return loaded

    def _evict_lru_until_fits(
        self,
        incoming_vram_mb: float,
    ) -> None:
        """Evict least-recently-used models until the incoming model fits."""
        while (
            self._resident
            and self.vram_used_mb() + incoming_vram_mb
            > self.vram_budget_mb
        ):
            victim_name = min(
                self._resident,
                key=lambda name: self._resident[name].last_used,
            )

            self.unload(victim_name)


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------


def _estimate_vram_mb(
    entry: ModelRegistryEntry,
    quantization_override: str | None,
) -> float:
    """
    Estimate VRAM usage when a model handle does not report it.

    These are deliberately conservative approximate values used only
    for the loader's eviction policy.
    """
    quantization = (
        quantization_override
        or entry.quantization
    )

    base_mb = {
        "none": 6000.0,
        "8bit": 3200.0,
        "4bit": 1800.0,
    }

    return base_mb.get(
        quantization,
        4000.0,
    )