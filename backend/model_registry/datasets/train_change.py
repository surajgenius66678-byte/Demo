import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

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
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

model = FCSiamDiff(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=13,
    classes=1,
)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

model.train()

epochs = 2

for epoch in range(epochs):
    total_loss = 0.0

    for step, batch in enumerate(loader):
        x = batch["image"]
        target = batch["mask"]

        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Step {step + 1}/{len(loader)} "
            f"Loss {loss.item():.4f}"
        )

    print(
        f"Epoch {epoch + 1} average loss: "
        f"{total_loss / len(loader):.4f}"
    )

torch.save(
    model.state_dict(),
    "checkpoints_change_oscd.pth",
)

print("TRAINING COMPLETE")