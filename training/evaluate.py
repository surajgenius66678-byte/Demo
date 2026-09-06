"""
Part 6 — evaluation.

Runs the fine-tuned adapter on test.jsonl — the split train.py never saw,
guaranteed by splits.py's leakage check — and writes a computed accuracy
number to eval_report.json. Nothing in this script hand-writes a metric: the
only way a number reaches that file is by being produced from actual model
output compared against ground truth, because both Part 4/5 downstream and
architecture.md's evidence-first principle depend on numbers that trace to
something real.

Metric: exact-match accuracy after light normalization (lowercase, strip
punctuation). This matches how RSVQA-style benchmarks are conventionally
scored, since its answers are short and categorical (yes/no, counts, land-
cover labels) rather than free-form text.

Usage:
    python evaluate.py --config config.yaml --adapter-dir ./runs/.../adapter
"""
import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def run_eval(cfg: dict, adapter_dir: str) -> dict:
    model_cfg = cfg["model"]
    processor = AutoProcessor.from_pretrained(adapter_dir)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    test_records = load_jsonl(Path(cfg["paths"]["splits_dir"]) / "test.jsonl")
    max_new_tokens = cfg["training"]["max_new_tokens_eval"]
    prompt_prefix = cfg["training"].get("prompt_prefix", "")

    correct = 0
    predictions = []
    with torch.no_grad():
        for rec in test_records:
            image = Image.open(rec["image_path"]).convert("RGB")
            prompt = prompt_prefix + rec["question"]
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            decoded = processor.decode(output_ids[0], skip_special_tokens=True)
            # PaliGemma echoes the prompt before the generated answer; strip it.
            prediction = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

            is_correct = normalize_answer(prediction) == normalize_answer(rec["answer"])
            correct += int(is_correct)
            predictions.append({
                "id": rec["id"],
                "question": rec["question"],
                "ground_truth": rec["answer"],
                "prediction": prediction,
                "correct": is_correct,
            })

    accuracy = correct / len(test_records) if test_records else 0.0
    return {
        "accuracy": accuracy,
        "n_test_examples": len(test_records),
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter-dir", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    results = run_eval(cfg, args.adapter_dir)
    print(f"Test accuracy: {results['accuracy']:.4f} over {results['n_test_examples']} examples")

    report_path = Path(cfg["paths"]["eval_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full report (incl. per-example predictions) written to {report_path}")


if __name__ == "__main__":
    main()
