"""
Validates checkpoint_metadata.json against the exact contract architecture.md
Section 3.6 defines. Run this after export_checkpoint.py, and again during the
Section 5 integration step before Part 4 ever reads the file.

Pure stdlib — no torch/transformers/peft needed, so this runs anywhere,
including inside the integration session that never trains anything itself.

Usage:
    python tests/test_checkpoint_contract.py path/to/checkpoint_metadata.json
"""
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = {
    "base_model": str,
    "adapter_type": str,
    "dataset": str,
    "eval": dict,
    "checkpoint_path": str,
}
VALID_ADAPTER_TYPES = {"LoRA", "QLoRA"}


def validate(metadata: dict, check_path_exists: bool = True) -> list[str]:
    errors = []

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in metadata:
            errors.append(f"missing required field: {field!r}")
            continue
        if not isinstance(metadata[field], expected_type):
            errors.append(
                f"{field!r} should be {expected_type.__name__}, "
                f"got {type(metadata[field]).__name__}"
            )

    if "adapter_type" in metadata and metadata["adapter_type"] not in VALID_ADAPTER_TYPES:
        errors.append(
            f"adapter_type must be one of {VALID_ADAPTER_TYPES}, got {metadata['adapter_type']!r}"
        )

    if "eval" in metadata and isinstance(metadata["eval"], dict):
        if not metadata["eval"]:
            errors.append("eval dict is empty — must contain at least one real metric")
        for k, v in metadata["eval"].items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errors.append(f"eval[{k!r}] must be numeric, got {type(v).__name__}")

    if check_path_exists and "checkpoint_path" in metadata and isinstance(metadata["checkpoint_path"], str):
        if not Path(metadata["checkpoint_path"]).exists():
            errors.append(f"checkpoint_path does not exist on disk: {metadata['checkpoint_path']}")

    return errors


def main():
    if len(sys.argv) != 2:
        print("usage: python tests/test_checkpoint_contract.py <checkpoint_metadata.json>")
        sys.exit(2)

    path = Path(sys.argv[1])
    with open(path) as f:
        metadata = json.load(f)

    errors = validate(metadata)
    if errors:
        print(f"FAIL — {len(errors)} contract violation(s) in {path}:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"PASS — {path} matches Section 3.6's contract.")


if __name__ == "__main__":
    main()
