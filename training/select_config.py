"""
Part 6 — GPU auto-detection.

Looks at the GPU actually available right now and recommends which
config.*.yaml profile matches its VRAM — so the same `run_pipeline.sh`,
unmodified, does the right thing whether it's run on a laptop RTX 3050 6GB,
a shared A100, or anything in between (Section 5's "run it, don't hand-tune
it per machine" spirit, applied to training hardware).

Two independent detection paths, tried in order, so this works whether or
not torch is installed yet:
  1. torch.cuda — authoritative once the environment is set up (matches
     exactly what training will actually see).
  2. `nvidia-smi` — works even before `pip install -r requirements.txt`,
     useful for a quick "what am I working with" check first.
No GPU found by either path -> prints a clear warning (a ~3B-parameter VLM
is not realistically trainable on CPU in hackathon time) and recommends the
lightest profile so a --dry-run at least demonstrates the pipeline is wired
correctly, rather than refusing to print anything.

Usage:
    python select_config.py            # human-readable report on stderr,
                                        # just the filename on stdout
    CONFIG=$(python select_config.py)  # capture just the filename, as
                                        # run_pipeline.sh does when no
                                        # config is given explicitly
"""
import subprocess
import sys

# (min_gb inclusive, max_gb exclusive) -> config file. Boundaries sit away
# from common card sizes (6, 8, 12, 16, 24, 40, 80 GB) so a GPU reporting
# slightly less than its nominal size (a few hundred MB is normally reserved
# by the driver/OS) doesn't flip it into the wrong tier.
TIERS = [
    (0, 10, "config.laptop-6gb.yaml"),
    (10, 20, "config.yaml"),
    (20, float("inf"), "config.a100.yaml"),
]


def _tier_for(vram_gb: float) -> str:
    for lo, hi, config in TIERS:
        if lo <= vram_gb < hi:
            return config
    return "config.yaml"  # unreachable given the ranges above; safe fallback


def detect_via_torch():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return []  # torch IS installed and definitively sees no CUDA GPU
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append((props.name, props.total_memory / (1024 ** 3)))
    return gpus


def detect_via_nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            name, mem_mib = parts
            try:
                gpus.append((name, float(mem_mib) / 1024))
            except ValueError:
                continue
    return gpus


def main():
    gpus = detect_via_torch()
    source = "torch.cuda"
    if gpus is None:
        gpus = detect_via_nvidia_smi()
        source = "nvidia-smi"

    if not gpus:
        print(
            "No CUDA GPU detected (checked: torch.cuda, nvidia-smi). "
            "Fine-tuning a ~3B-parameter VLM on CPU is not realistic in "
            "hackathon time. Recommending the lightest profile anyway so "
            "`--dry-run` still demonstrates the pipeline is wired correctly; "
            "the real run needs an actual GPU, laptop or cloud/external.",
            file=sys.stderr,
        )
        print("config.laptop-6gb.yaml")
        return

    name, vram_gb = max(gpus, key=lambda g: g[1])  # biggest GPU if there are several
    config = _tier_for(vram_gb)
    print(f"Detected via {source}: {name} ({vram_gb:.1f} GB VRAM)", file=sys.stderr)
    if len(gpus) > 1:
        print(f"  ({len(gpus)} GPUs visible; picked the largest — "
              f"multi-GPU training itself isn't set up in train.py, "
              f"just single-device)", file=sys.stderr)
    print(f"-> recommending {config}", file=sys.stderr)
    print(config)


if __name__ == "__main__":
    main()
