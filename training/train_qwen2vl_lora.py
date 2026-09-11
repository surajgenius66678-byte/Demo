from pathlib import Path
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model


MODEL_ID = "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct"
DATA_ROOT = Path("data/rsvqa_lora")
METADATA = DATA_ROOT / "metadata.jsonl"
OUTPUT_DIR = Path("checkpoints/rs_vlm_lora")


class RSVQALoRADataset(Dataset):
    def __init__(self, metadata_path, processor):
        self.processor = processor
        self.records = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        image = Image.open(r["image_path"]).convert("RGB")
        question = r["question"]
        answer = str(r["answer"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Answer the following remote-sensing question "
                            "using only the visual evidence in the image.\n"
                            f"Question: {question}"
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ],
            },
        ]

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Answer the following remote-sensing question "
                            "using only the visual evidence in the image.\n"
                            f"Question: {question}"
                        ),
                    },
                ],
            }
        ]

        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[full_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_length] = -100

        # Ignore padding.
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"][0] == 0] = -100

        inputs = {
            k: v[0] if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v
            for k, v in inputs.items()
        }

        inputs["labels"] = labels

        return inputs


def main():
    print("=" * 70)
    print("SatQuery AI — Remote-Sensing Qwen2-VL LoRA Adaptation")
    print("=" * 70)

    if not METADATA.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {METADATA}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device: CPU")
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {METADATA}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading Qwen2-VL...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    print("Applying LoRA...")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = RSVQALoRADataset(METADATA, processor)

    print(f"Dataset samples: {len(dataset)}")

    # CPU-safe first training configuration.
    #
    # IMPORTANT:
    # This is intentionally a small smoke adaptation first.
    # Once it succeeds, we can increase the number of samples.
    max_samples = min(20, len(dataset))

    class LimitedDataset(Dataset):
        def __init__(self, base, n):
            self.base = base
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return self.base[idx]

    train_dataset = LimitedDataset(dataset, max_samples)

    print(f"Smoke-training samples: {len(train_dataset)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5,
    )

    model.train()

    for epoch in range(1):
        print(f"\nEpoch {epoch + 1}/1")

        total_loss = 0.0

        for i in range(len(train_dataset)):
            batch = train_dataset[i]

            batch = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() >= 1 else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(
                f"  sample {i + 1}/{len(train_dataset)} "
                f"loss={loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_dataset)

        print(f"Average loss: {avg_loss:.4f}")

    adapter_path = OUTPUT_DIR / "adapter"

    print("\nSaving LoRA adapter...")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)

    metadata = {
        "base_model": MODEL_ID,
        "adapter_type": "LoRA",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        "rank": 4,
        "alpha": 8,
        "dropout": 0.05,
        "training_dataset": "dmarsili/RSVQA-LR-2k",
        "local_dataset_samples": len(dataset),
        "smoke_training_samples": len(train_dataset),
        "epochs": 1,
        "device": "cpu",
        "average_training_loss": avg_loss,
    }

    (OUTPUT_DIR / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("ADAPTATION SMOKE TEST COMPLETE")
    print("=" * 70)
    print(f"Adapter: {adapter_path}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Metadata: {OUTPUT_DIR / 'training_metadata.json'}")


if __name__ == "__main__":
    main()
