import json
import numpy as np
import os

episode_dir = "parsed_demo_0"

state = np.load(os.path.join(episode_dir, "state.npy"))
actions = np.load(os.path.join(episode_dir, "actions_23.npy"))

with open(os.path.join(episode_dir, "manifest.json")) as f:
    records = json.load(f)

print("records:", len(records))
print("state:", state.shape)
print("actions:", actions.shape)

for idx in [0, len(records)//2, len(records)-1]:
    r = records[idx]
    print("\nrecord", idx)
    print(" image_path:", r["image_path"], "exists=", os.path.exists(r["image_path"]))
    print(" instruction:", r["instruction"])
    print(" state first 8:", state[r["state_index"]][:8].tolist())
    print(" action:", actions[r["action_index"]].tolist())