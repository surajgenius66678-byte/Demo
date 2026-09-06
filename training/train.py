"""
Part 6 — LoRA / QLoRA fine-tuning.

Fine-tunes the configured base VLM (default: PaliGemma, see config.yaml) on
train.jsonl using PEFT, validating on val.jsonl. Adapter type and every
hyperparameter come from config.yaml — swapping LoRA vs QLoRA, rank, target
modules, or the base model itself is a config edit, not a code change.

Fine-tuning pattern (prompt as `text`, target answer as `suffix`) follows the
transformers docs' PaliGemma fine-tuning example. QLoRA-specific steps
(prepare_model_for_kbit_training, enabling input grads before gradient
checkpointing) are included because skipping them is a common way for a
QLoRA + gradient-checkpointing run to silently fail to backprop.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --dry-run   # see below

--dry-run: runs on a handful of real records for a handful of steps —
same model, same quantization, same batch size, same collate_fn — then
exits before saving anything. It exists to answer one question in minutes
on your own machine, before a real multi-hour/GPU-hours run: does this
config actually run a full forward+backward step without OOMing or hitting
a data/shape bug, on THIS GPU. It does not tell you whether the model will
converge to a good answer — only whether the run is mechanically sound.
Prints peak GPU memory used, which is the number to compare against your
card's total VRAM when choosing a config profile (see the config.*.yaml
files and README's "Which config to use" section).

Resumable by default: if training.output_dir already has a checkpoint-N
from a previous (possibly interrupted) run, training continues from there
instead of starting over — see find_latest_checkpoint() below. Pass
--no-resume to start fresh anyway (needed after a config change the old
checkpoint isn't compatible with, e.g. a different base model or rank).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Returns the highest-numbered checkpoint-N directory under output_dir,
    or None if there isn't one. Used to auto-resume an interrupted run
    instead of starting over — Ctrl-C, a killed process, a shared/pre-emptible
    GPU box, or a laptop that went to sleep mid-run should cost at most the
    steps since the last save_steps checkpoint, not the whole run."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    checkpoints = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-")[-1])
            except ValueError:
                continue
            checkpoints.append((step, child))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda pair: pair[0])
    return str(checkpoints[-1][1])


def build_model_and_processor(cfg: dict):
    model_cfg = cfg["model"]
    adapter_cfg = cfg["adapter"]

    processor = AutoProcessor.from_pretrained(model_cfg["processor"])

    quantization_config = None
    if adapter_cfg["type"] == "QLoRA":
        bits = adapter_cfg.get("quantization_bits", 4)
        if bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif bits == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"adapter.quantization_bits must be 4 or 8, got {bits}")
    elif adapter_cfg["type"] != "LoRA":
        raise ValueError(f"adapter.type must be 'LoRA' or 'QLoRA', got {adapter_cfg['type']!r}")

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_cfg["base_model"],
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if quantization_config is not None:
        # Standard QLoRA prep: casts norm layers to fp32, preps the model so
        # gradient checkpointing works with a frozen quantized base.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
        )

    lora_config = LoraConfig(
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["alpha"],
        lora_dropout=adapter_cfg["dropout"],
        target_modules=adapter_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if cfg["training"]["gradient_checkpointing"]:
        model.enable_input_require_grads()  # required or backward silently no-ops on a frozen base

    model.print_trainable_parameters()
    return model, processor


def build_collate_fn(processor, prompt_prefix: str):
    def collate_fn(batch):
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        prompts = [prompt_prefix + r["question"] for r in batch]
        answers = [r["answer"] for r in batch]

        return processor(
            text=prompts,
            images=images,
            suffix=answers,
            return_tensors="pt",
            padding="longest",
        )

    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run a handful of real steps on a handful of real records, then "
             "exit without saving. Verifies the config/environment/GPU combination "
             "actually works before spending a full run's worth of GPU time.",
    )
    parser.add_argument(
        "--dry-run-steps", type=int, default=3,
        help="Optimizer steps to run under --dry-run (default: 3).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing checkpoint under training.output_dir and start "
             "fresh. Default behavior auto-resumes from the latest checkpoint if "
             "one exists (e.g. after Ctrl-C, a killed process, or a sleeping "
             "laptop) — use this after a config change that isn't compatible with "
             "the old run (e.g. a different base model or adapter rank).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    splits_dir = Path(cfg["paths"]["splits_dir"])
    train_records = load_jsonl(splits_dir / "train.jsonl")
    val_records = load_jsonl(splits_dir / "val.jsonl")

    train_cfg = cfg["training"]
    if args.dry_run:
        # Real records, real model, real batch size — just few enough of each
        # that this finishes in minutes on any GPU it'll run on at all.
        need = max(4, train_cfg["per_device_batch_size"] * (args.dry_run_steps + 1))
        if len(train_records) < need:
            print(
                f"[dry-run] only {len(train_records)} train records available, "
                f"want {need} for {args.dry_run_steps} steps at batch size "
                f"{train_cfg['per_device_batch_size']} — will just do fewer "
                "internal epochs over what exists rather than fail.",
                file=sys.stderr,
            )
        val_records = val_records[: max(2, train_cfg["per_device_batch_size"])]
        print(f"[dry-run] {len(train_records)} train / {len(val_records)} val "
              f"records available, running {args.dry_run_steps} step(s)")
    else:
        print(f"train: {len(train_records)}  val: {len(val_records)}")

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    model, processor = build_model_and_processor(cfg)
    collate_fn = build_collate_fn(processor, train_cfg.get("prompt_prefix", ""))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output_dir = train_cfg["output_dir"] if not args.dry_run else f"{train_cfg['output_dir']}-dryrun"
    args_tr = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        max_steps=args.dry_run_steps if args.dry_run else -1,  # overrides num_train_epochs when >0
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=0 if args.dry_run else train_cfg["warmup_steps"],
        logging_steps=1 if args.dry_run else train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_strategy="no" if args.dry_run else "steps",  # transformers >= 4.46; "evaluation_strategy" on older versions
        eval_steps=None if args.dry_run else train_cfg["save_steps"],
        bf16=True,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg.get("optimizer", "adamw_torch"),  # e.g. "paged_adamw_8bit" on tight-VRAM profiles
        report_to=[],                     # no experiment tracker wired up — hackathon default
        remove_unused_columns=False,      # collate_fn needs the raw dict fields, not model.forward()'s signature
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    resume_path = None
    if not args.dry_run and not args.no_resume:
        resume_path = find_latest_checkpoint(output_dir)
    if resume_path:
        print(f"Found existing checkpoint {resume_path!r} — resuming from there "
              f"(pass --no-resume to start fresh instead).")
    elif not args.dry_run:
        print(f"No existing checkpoint under {output_dir!r} — starting fresh.")

    trainer.train(resume_from_checkpoint=resume_path)

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak GPU memory allocated: {peak_gb:.2f} GB "
              f"(compare against your card's total VRAM)")

    if args.dry_run:
        print(
            f"\n[dry-run] {args.dry_run_steps} step(s) completed without error "
            f"at batch size {train_cfg['per_device_batch_size']}. Nothing was "
            f"saved (output under {output_dir} is safe to delete). This checks "
            "the run is mechanically sound on this GPU — it says nothing about "
            "whether the model will actually learn; that's what the real run "
            "plus evaluate.py is for."
        )
        return

    adapter_dir = str(Path(output_dir) / "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
