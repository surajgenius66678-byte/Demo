import torch
import torch.nn as nn
from torch.optim import Adam

from backend.model_registry.datasets.oscd_local import LocalOSCDDataset
from torchgeo.models import FCSiamDiff


IMAGES_ROOT = (
    r"E:\Dataset\Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images"
    r"\Onera Satellite Change Detection dataset - Images"
)

LABELS_ROOT = (
    r"E:\Dataset\OSCD100_Train_Labels"
    r"\Onera Satellite Change Detection dataset - Train Labels"
)


dataset = LocalOSCDDataset(
    images_root=IMAGES_ROOT,
    labels_root=LABELS_ROOT,
    split="train",
)

sample = dataset[0]

x = sample["image"].unsqueeze(0)
target = sample["mask"].unsqueeze(0)

model = FCSiamDiff(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=13,
    classes=1,
)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

model.train()

optimizer.zero_grad()

output = model(x)

loss = criterion(output, target)

loss.backward()

optimizer.step()

print("Training smoke test: PASS")
print("Input:", x.shape)
print("Target:", target.shape)
print("Output:", output.shape)
print("Loss:", float(loss))