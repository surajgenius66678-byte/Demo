from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection


class GroundingDINOModel:
    """
    Real text-guided grounding specialist backed by Grounding DINO.

    Contract expected by GroundingAdapter:
        ground(image_path, query) -> list[dict]
    """

    MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID
        )

        self.model = GroundingDinoForObjectDetection.from_pretrained(
            self.MODEL_ID
        )

        self.model.to(self.device)
        self.model.eval()

    def ground(self, image_path: str, query: str):
        if not query or not query.strip():
            return []

        image = Image.open(Path(image_path)).convert("RGB")

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
        labels = results.get("text_labels", results.get("labels", []))

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
