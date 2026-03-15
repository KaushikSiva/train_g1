import os
import json
import numpy as np

EPISODE_DIR = "parsed_demo_0"
FRAMES_DIR = os.path.join(EPISODE_DIR, "frames")

state = np.load(os.path.join(EPISODE_DIR, "state.npy"))
actions = np.load(os.path.join(EPISODE_DIR, "actions_23.npy"))

instruction = "Organize the beverages neatly on the table."

num_steps = state.shape[0]
assert actions.shape[0] == num_steps, (actions.shape[0], num_steps)

records = []
for t in range(num_steps):
    frame_path = os.path.join(FRAMES_DIR, f"frame_{t:06d}.jpg")
    if not os.path.exists(frame_path):
        raise FileNotFoundError(frame_path)

    records.append({
        "timestep": t,
        "image_path": frame_path,
        "instruction": instruction,
        "state_index": t,
        "action_index": t,
    })

with open(os.path.join(EPISODE_DIR, "manifest.json"), "w") as f:
    json.dump(records, f, indent=2)

print("saved manifest with", len(records), "records")
print("first record:", records[0])
print("last record:", records[-1])