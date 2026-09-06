"""Exceptions raised by Part 4 — Model Registry & Specialist Inference Engine."""


class ModelRegistryError(Exception):
    """Base class for every error this part raises."""


class ModelNotFoundError(ModelRegistryError):
    """No registry entry can serve the requested task/modality combination."""


class ModelLoadError(ModelRegistryError):
    """A model failed to load (bad checkpoint, missing dependency, etc.)."""


class InferenceOOMError(ModelRegistryError):
    """
    Reserved for a truly unrecoverable load-time failure (e.g. a model that
    cannot even be loaded at its lightest quantization tier). Ordinary
    inference-time OOM is handled internally by the retry ladder in
    inference.py and surfaces as a warning on a still schema-valid Evidence
    instead of this exception — see run_inference()'s docstring.
    """
