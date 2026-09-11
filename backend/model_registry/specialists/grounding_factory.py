from backend.model_registry.specialists.grounding_dino import GroundingDINOModel


def load_grounding_model(entry, device, quantization_override):
    """
    Factory for the SatQuery Grounding DINO specialist.
    """
    return GroundingDINOModel(device=device)