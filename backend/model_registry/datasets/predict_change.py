from pathlib import Path

import numpy as np
import torch
import rasterio
from PIL import Image

from torchgeo.models import FCSiamDiff


BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

ROOT = Path(
    r"E:\Dataset\Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images"
    r"\Onera Satellite Change Detection dataset - Images"
)

LABEL_ROOT = Path(
    r"E:\Dataset\OSCD100_Test_Labels"
    r"\Onera Satellite Change Detection dataset - Test Labels"
)

CHECKPOINT = "checkpoints_change_oscd.pth"


def load_image(folder):
    bands = []

    for band in BANDS:
        with rasterio.open(folder / f"{band}.tif") as src:
            x = src.read(1).astype(np.float32)

        low = np.percentile(x, 2)
        high = np.percentile(x, 98)

        if high > low:
            x = np.clip((x - low) / (high - low), 0, 1)
        else:
            x = np.zeros_like(x)

        bands.append(x)

    return np.stack(bands, axis=0)


model = FCSiamDiff(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=13,
    classes=1,
)

model.load_state_dict(
    torch.load(CHECKPOINT, map_location="cpu")
)

model.eval()

tp = fp = fn = tn = 0

for i in range(20):
    name = f"test_{i:03d}"

    before = torch.from_numpy(
        load_image(ROOT / name / "imgs_1_rect")
    )

    after = torch.from_numpy(
        load_image(ROOT / name / "imgs_2_rect")
    )

    x = torch.stack([before, after]).unsqueeze(0)

    label_path = LABEL_ROOT / name / "cm" / "cm.png"
    target = np.array(Image.open(label_path)) > 0
    target = target.reshape(-1)

    with torch.no_grad():
        logits = model(x)
        prediction = torch.sigmoid(logits)[0, 0].numpy() >= 0.5

    prediction = prediction.reshape(-1)

    tp += np.sum(prediction & target)
    fp += np.sum(prediction & ~target)
    fn += np.sum(~prediction & target)
    tn += np.sum(~prediction & ~target)

    print(f"{name}: done")

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall
    else 0
)
iou = tp / (tp + fp + fn) if tp + fp + fn else 0

print()
print("OSCD TEST RESULTS")
print("-----------------")
print(f"TP:        {tp}")
print(f"FP:        {fp}")
print(f"FN:        {fn}")
print(f"TN:        {tn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"IoU:       {iou:.4f}")