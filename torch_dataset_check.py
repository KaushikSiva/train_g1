import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

EPISODE_DIR = "parsed_demo_0"

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

        return {
            "image": image,
            "state": state,
            "action": action,
            "instruction": rec["instruction"],
            "timestep": rec["timestep"],
        }

def collate_fn(batch):
    return {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "state": torch.stack([x["state"] for x in batch], dim=0),
        "action": torch.stack([x["action"] for x in batch], dim=0),
        "instruction": [x["instruction"] for x in batch],
        "timestep": torch.tensor([x["timestep"] for x in batch], dtype=torch.long),
    }

dataset = BevOrgDataset(EPISODE_DIR)
loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

batch = next(iter(loader))

print("dataset len:", len(dataset))
print("image batch shape:", tuple(batch["image"].shape))
print("state batch shape:", tuple(batch["state"].shape))
print("action batch shape:", tuple(batch["action"].shape))
print("instructions:", batch["instruction"])
print("timesteps:", batch["timestep"].tolist())
print("first action[0][:8]:", batch["action"][0][:8].tolist())
