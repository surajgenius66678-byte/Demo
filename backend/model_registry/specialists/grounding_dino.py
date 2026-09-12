from pathlib import Path
import os

import torch
import numpy as np
import rasterio
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection


class GroundingDINOModel:
    """
    Real text-guided grounding specialist backed by Grounding DINO.

    Uses the locally cached Hugging Face model when available so inference
    does not require an internet connection.

    Contract expected by GroundingAdapter:
        ground(image_path, query) -> list[dict]
    """

    MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        # Optional explicit override:
        # SATQUERY_GROUNDING_MODEL_PATH=<local snapshot directory>
        explicit_path = os.environ.get("SATQUERY_GROUNDING_MODEL_PATH")

        if explicit_path:
            model_path = Path(explicit_path).expanduser()
        else:
            hf_cache = (
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--IDEA-Research--grounding-dino-base"
            )

            snapshots_dir = hf_cache / "snapshots"

            if not snapshots_dir.exists():
                raise FileNotFoundError(
                    "Grounding DINO is not available in the local Hugging Face "
                    f"cache: {snapshots_dir}"
                )

            snapshots = [
                path for path in snapshots_dir.iterdir()
                if path.is_dir()
            ]

            if not snapshots:
                raise FileNotFoundError(
                    "No Grounding DINO snapshot was found in: "
                    f"{snapshots_dir}"
                )

            # Use the newest cached snapshot.
            model_path = max(
                snapshots,
                key=lambda path: path.stat().st_mtime,
            )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Grounding DINO model path does not exist: {model_path}"
            )

        config_path = model_path / "config.json"
        weights_path = model_path / "model.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Grounding DINO config.json is missing: {config_path}"
            )

        # Hugging Face cache snapshots can contain symlinks/references whose
        # displayed size is 0 bytes while the real blob lives in `blobs`.
        # Transformers can resolve those references normally.
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Grounding DINO model weights are missing: {weights_path}"
            )

        self.model_path = model_path

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )

        self.model = GroundingDinoForObjectDetection.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

    def ground(self, image_path: str, query: str):
        if not query or not query.strip():
            return []

        with rasterio.open(image_path) as src:
            raster = src.read()

        if raster.shape[0] >= 3:
            rgb = raster[[0, 1, 2], :, :]
        else:
            raise ValueError(
                f"Grounding DINO requires at least 3 bands, got {raster.shape[0]}"
            )

        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255.0).astype(np.uint8)

        image = Image.fromarray(rgb, mode="RGB")

        # Grounding DINO works best with phrase-style prompts.
        text = query.strip()
        if not text.endswith("."):
            text += "."

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [[image.height, image.width]],
            device=self.device,
        )

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.15,
            text_threshold=0.25,
            target_sizes=target_sizes,
        )[0]

        detections = []

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get(
            "text_labels",
            results.get("labels", []),
        )

        for box, score, label in zip(boxes, scores, labels):
            box = box.detach().cpu().tolist()
            score = float(score.detach().cpu())

            detections.append(
                {
                    "label": str(label),
                    "box_px": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    "mask_rle": None,
                    "score": score,
                }
            )

        return detections