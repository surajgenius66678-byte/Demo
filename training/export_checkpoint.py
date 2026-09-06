"""
Part 6 — checkpoint export.

Writes checkpoint_metadata.json in the exact shape architecture.md Section
3.6 defines as the contract Part 6 owes Part 4:

    {
      "base_model": "string",
      "adapter_type": "LoRA | QLoRA",
      "dataset": "string",
      "eval": {"metric_name": 0.0},
      "checkpoint_path": "string"
    }

This is the ONLY coupling point with Part 4 — file-based, per Section 3.6's
"Depends on" line; Part 4 never imports anything from this folder. Keep this
file's shape stable even if the rest of the pipeline changes underneath it.

Refuses to run before evaluate.py has produced a real eval_report.json, so
this can never emit a placeholder or invented number into the contract.

Usage:
    python export_checkpoint.py --config config.yaml --adapter-dir ./runs/.../adapter
"""
import argparse
import json
from pathlib import Path

import yaml

VALID_ADAPTER_TYPES = {"LoRA", "QLoRA"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_metadata(cfg: dict, adapter_dir: str, eval_report: dict, metric_name: str) -> dict:
    adapter_type = cfg["adapter"]["type"]
    if adapter_type not in VALID_ADAPTER_TYPES:
        raise ValueError(
            f"adapter.type in config.yaml must be one of {VALID_ADAPTER_TYPES}, got {adapter_type!r}"
        )

    return {
        "base_model": cfg["model"]["base_model"],
        "adapter_type": adapter_type,
        "dataset": cfg["dataset"]["name"],
        "eval": {metric_name: eval_report["accuracy"]},
        # Points at the PEFT adapter directory (adapter_config.json +
        # adapter weights), not a merged full model. Part 4's adapter loads
        # it with: PeftModel.from_pretrained(base_model, checkpoint_path).
        # Call model.merge_and_unload() first if Part 4 needs a single
        # merged model instead.
        "checkpoint_path": str(Path(adapter_dir).resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter-dir", required=True,
                         help="Directory train.py saved the PEFT adapter to.")
    parser.add_argument("--eval-metric-name", default="accuracy",
                         help="Key name for the eval metric, e.g. 'accuracy' or 'bleu'.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    eval_report_path = Path(cfg["paths"]["eval_report_path"])
    if not eval_report_path.exists():
        raise FileNotFoundError(
            f"{eval_report_path} not found — run evaluate.py before "
            "export_checkpoint.py. checkpoint_metadata.json must carry a "
            "real computed number, never a placeholder."
        )
    with open(eval_report_path) as f:
        eval_report = json.load(f)

    metadata = build_metadata(cfg, args.adapter_dir, eval_report, args.eval_metric_name)

    out_path = Path(cfg["paths"]["checkpoint_metadata_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"checkpoint_metadata.json written to {out_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
