from backend.model_registry.specialists.croma_fusion import CROMAFusionModel


def load_croma_fusion_model(
    entry,
    device,
    quantization_override,
):
    return CROMAFusionModel(device=device)