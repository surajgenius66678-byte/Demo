from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]


class LocalOSCDDataset(Dataset):
    def __init__(
        self,
        images_root: str,
        labels_root: str,
        split: str = "train",
    ):
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.split = split

        split_root = self.images_root
        label_root = self.labels_root
        self.samples = []

        for pair_dir in sorted(split_root.glob(f"{split}_*")):
            label_dir = label_root / pair_dir.name / "cm"
            label_path = label_dir / "cm.png"

            if not label_path.exists():
                continue

            before_dir = pair_dir / "imgs_1_rect"
            after_dir = pair_dir / "imgs_2_rect"

            if not before_dir.exists() or not after_dir.exists():
                continue

            self.samples.append(
                {
                    "id": pair_dir.name,
                    "before": before_dir,
                    "after": after_dir,
                    "mask": label_path,
                }
            )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, folder: Path) -> torch.Tensor:
        import rasterio

        bands = []

        for band in BANDS:
            path = folder / f"{band}.tif"

            with rasterio.open(path) as src:
                array = src.read(1).astype(np.float32)

            bands.append(array)

        image = np.stack(bands, axis=0)

        # Per-band normalization to [0, 1].
        for i in range(image.shape[0]):
            band = image[i]
            low = np.percentile(band, 2)
            high = np.percentile(band, 98)

            if high > low:
                image[i] = np.clip(
                    (band - low) / (high - low),
                    0.0,
                    1.0,
                )
            else:
                image[i] = 0.0

        return torch.from_numpy(image)

    def _load_mask(self, path: Path) -> torch.Tensor:
        from PIL import Image

        mask = np.array(Image.open(path), dtype=np.float32)

        mask = (mask > 0).astype(np.float32)

        return torch.from_numpy(mask).unsqueeze(0)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        before = self._load_image(sample["before"])
        after = self._load_image(sample["after"])
        mask = self._load_mask(sample["mask"])

        image = torch.stack([before, after], dim=0)

        return {
            "image": image,
            "mask": mask,
            "id": sample["id"],
        }