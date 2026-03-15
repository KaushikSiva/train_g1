import os
import json
import h5py
import numpy as np

H5_PATH = "data/lightwheel_robocasa_tasks_g1_wbc/BeverageOrganization/BeverageOrganization_1763692193809507/trajectories.hdf5"
DEMO_KEY = "data/demo_0"
OUT_DIR = "parsed_demo_0"

os.makedirs(OUT_DIR, exist_ok=True)

with h5py.File(H5_PATH, "r") as f:
    joint_pos = f[f"{DEMO_KEY}/states/articulation/robot/joint_position"][:]
    joint_vel = f[f"{DEMO_KEY}/states/articulation/robot/joint_velocity"][:]
    root_pose = f[f"{DEMO_KEY}/states/articulation/robot/root_pose"][:]
    root_vel = f[f"{DEMO_KEY}/states/articulation/robot/root_velocity"][:]
    ee_pose = f[f"{DEMO_KEY}/obs/ee_pose"][:]

    actions_23 = f[f"{DEMO_KEY}/obs/actions"][:]
    processed_actions_55 = f[f"{DEMO_KEY}/processed_actions"][:]
    joint_pos_target_172 = f[f"{DEMO_KEY}/joint_targets/joint_pos_target"][:]

    checkpoint_frame_idx = f[f"{DEMO_KEY}/obs/raw_input/internal_state/last_checkpoint_frame_idx"][:]

state = np.concatenate(
    [
        joint_pos,
        joint_vel,
        root_pose,
        root_vel,
        ee_pose.reshape(ee_pose.shape[0], -1),
    ],
    axis=1,
)

np.save(os.path.join(OUT_DIR, "state.npy"), state)
np.save(os.path.join(OUT_DIR, "joint_pos.npy"), joint_pos)
np.save(os.path.join(OUT_DIR, "joint_vel.npy"), joint_vel)
np.save(os.path.join(OUT_DIR, "root_pose.npy"), root_pose)
np.save(os.path.join(OUT_DIR, "root_vel.npy"), root_vel)
np.save(os.path.join(OUT_DIR, "ee_pose.npy"), ee_pose)

np.save(os.path.join(OUT_DIR, "actions_23.npy"), actions_23)
np.save(os.path.join(OUT_DIR, "processed_actions_55.npy"), processed_actions_55)
np.save(os.path.join(OUT_DIR, "joint_pos_target_172.npy"), joint_pos_target_172)
np.save(os.path.join(OUT_DIR, "checkpoint_frame_idx.npy"), checkpoint_frame_idx)

meta = {
    "demo_key": DEMO_KEY,
    "timesteps_state": int(state.shape[0]),
    "timesteps_actions_23": int(actions_23.shape[0]),
    "timesteps_processed_actions_55": int(processed_actions_55.shape[0]),
    "timesteps_joint_pos_target_172": int(joint_pos_target_172.shape[0]),
    "timesteps_checkpoint_frame_idx": int(checkpoint_frame_idx.shape[0]),
    "state_dim": int(state.shape[1]),
    "actions_23_dim": int(actions_23.shape[1]),
    "processed_actions_55_dim": int(processed_actions_55.shape[1]),
    "joint_pos_target_172_dim": int(joint_pos_target_172.shape[1]),
    "instruction": "Organize the beverages neatly on the table."
}

with open(os.path.join(OUT_DIR, "meta.json"), "w") as fp:
    json.dump(meta, fp, indent=2)

print("Saved parsed demo to:", OUT_DIR)
print("state shape:", state.shape)
print("actions_23 shape:", actions_23.shape)
print("processed_actions_55 shape:", processed_actions_55.shape)
print("joint_pos_target_172 shape:", joint_pos_target_172.shape)
print("checkpoint_frame_idx shape:", checkpoint_frame_idx.shape)
