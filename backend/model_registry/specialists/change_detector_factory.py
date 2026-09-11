from pathlib import Path

from backend.model_registry.specialists.change_detector import OSCDChangeDetector


def load_oscd_change_detector(entry, device, quantization_override):
    """
    Factory for the SatQuery OSCD bi-temporal change detector.
    """

    checkpoint_path = Path(__file__).resolve().parents[3] / "checkpoints_change_oscd.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"OSCD change checkpoint not found: {checkpoint_path}"
        )

    return OSCDChangeDetector(
        checkpoint_path=str(checkpoint_path),
        device=device,
    )
