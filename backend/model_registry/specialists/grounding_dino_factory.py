from backend.model_registry.specialists.grounding_dino import GroundingDINOModel


def load_grounding_dino(entry, device, quantization_override):
    """
    Factory for the real Grounding DINO text-guided grounding model.
    """

    return GroundingDINOModel(
        device=device,
    )
