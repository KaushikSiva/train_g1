import os
import json
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models


EPISODE_DIR = "parsed_demo_0"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class BevOrgDataset(Dataset):
    def __init__(self, episode_dir):
        self.episode_dir = episode_dir
        with open(os.path.join(episode_dir, "manifest.json")) as f:
            self.records = json.load(f)

        self.state = np.load(os.path.join(episode_dir, "state.npy")).astype(np.float32)
        self.action = np.load(os.path.join(episode_dir, "actions_23.npy")).astype(np.float32)

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        image = Image.open(rec["image_path"]).convert("RGB")
        image = self.image_transform(image)

        state = torch.from_numpy(self.state[rec["state_index"]])
        action = torch.from_numpy(self.action[rec["action_index"]])

        return image, state, action


class BaselinePolicy(nn.Module):
    def __init__(self, state_dim=113, action_dim=23):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, image, state):
        img_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state)
        feat = torch.cat([img_feat, state_feat], dim=1)
        return self.head(feat)


def make_loaders():
    ds = BevOrgDataset(EPISODE_DIR)

    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train

    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for image, state, action in loader:
            image = image.to(device)
            state = state.to(device)
            action = action.to(device)

            pred = model(image, state)
            loss = criterion(pred, action)

            bs = image.size(0)
            total_loss += loss.item() * bs
            total_count += bs

    return total_loss / max(total_count, 1)


def main():
    print("Using device:", DEVICE)

    train_loader, val_loader = make_loaders()
    model = BaselinePolicy().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for step, (image, state, action) in enumerate(train_loader, start=1):
            image = image.to(DEVICE)
            state = state.to(DEVICE)
            action = action.to(DEVICE)

            optimizer.zero_grad()
            pred = model(image, state)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()

            bs = image.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            if step % 20 == 0:
                print(f"epoch {epoch} step {step} train_loss {loss.item():.6f}")

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)

        print(f"epoch {epoch} done | train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": 113,
                    "action_dim": 23,
                },
                "baseline_policy.pt",
            )
            print("saved best model -> baseline_policy.pt")

    print("best_val:", best_val)


if __name__ == "__main__":
    main()
