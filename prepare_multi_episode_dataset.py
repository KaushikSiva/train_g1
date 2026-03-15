import os
import json
import glob
import h5py
import cv2
import numpy as np

TASK_ROOT = "data/lightwheel_robocasa_tasks_g1_wbc/BeverageOrganization"
OUT_ROOT = "prepared_bevorg_dataset"
INSTRUCTION = "Organize the beverages neatly on the table."

os.makedirs(OUT_ROOT, exist_ok=True)


def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if existing:
        return len(existing)

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        idx += 1

    cap.release()
    return idx


def load_episode_tensors(h5_path, demo_key="data/demo_0"):
    with h5py.File(h5_path, "r") as f:
        joint_pos = f[f"{demo_key}/states/articulation/robot/joint_position"][:]
        joint_vel = f[f"{demo_key}/states/articulation/robot/joint_velocity"][:]
        root_pose = f[f"{demo_key}/states/articulation/robot/root_pose"][:]
        root_vel = f[f"{demo_key}/states/articulation/robot/root_velocity"][:]
        ee_pose = f[f"{demo_key}/obs/ee_pose"][:]
        actions_23 = f[f"{demo_key}/obs/actions"][:]

    state = np.concatenate(
        [
            joint_pos,
            joint_vel,
            root_pose,
            root_vel,
            ee_pose.reshape(ee_pose.shape[0], -1),
        ],
        axis=1,
    ).astype(np.float32)

    actions_23 = actions_23.astype(np.float32)

    return state, actions_23


def inspect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def process_episode(ep_dir, episode_idx):
    episode_name = os.path.basename(ep_dir.rstrip("/"))
    h5_path = os.path.join(ep_dir, "trajectories.hdf5")
    mp4_path = os.path.join(ep_dir, "isaac_replay_state_product.mp4")

    if not os.path.exists(h5_path):
        print(f"[skip] missing h5: {episode_name}")
        return None
    if not os.path.exists(mp4_path):
        print(f"[skip] missing mp4: {episode_name}")
        return None

    print(f"\n[episode] {episode_name}")

    state, actions = load_episode_tensors(h5_path)
    video_info = inspect_video(mp4_path)

    num_steps = state.shape[0]
    num_actions = actions.shape[0]
    num_frames_video = video_info["frame_count"]

    if num_steps != num_actions:
        print(f"[skip] state/action mismatch for {episode_name}: {num_steps} vs {num_actions}")
        return None

    episode_out_dir = os.path.join(OUT_ROOT, f"episode_{episode_idx:04d}")
    frames_dir = os.path.join(episode_out_dir, "frames")
    os.makedirs(episode_out_dir, exist_ok=True)

    extracted_frame_count = extract_frames(mp4_path, frames_dir)

    if extracted_frame_count != num_steps:
        print(
            f"[warn] frame count mismatch in {episode_name}: "
            f"video={num_frames_video}, extracted={extracted_frame_count}, steps={num_steps}"
        )

    usable_steps = min(num_steps, extracted_frame_count)

    state = state[:usable_steps]
    actions = actions[:usable_steps]

    np.save(os.path.join(episode_out_dir, "state.npy"), state)
    np.save(os.path.join(episode_out_dir, "actions_23.npy"), actions)

    manifest = []
    for t in range(usable_steps):
        frame_path = os.path.join(frames_dir, f"frame_{t:06d}.jpg")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(frame_path)

        manifest.append({
            "episode_name": episode_name,
            "episode_local_id": f"episode_{episode_idx:04d}",
            "timestep": t,
            "image_path": frame_path,
            "instruction": INSTRUCTION,
            "state_index": t,
            "action_index": t,
        })

    with open(os.path.join(episode_out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    meta = {
        "source_episode_dir": ep_dir,
        "episode_name": episode_name,
        "episode_local_id": f"episode_{episode_idx:04d}",
        "timesteps": usable_steps,
        "state_dim": int(state.shape[1]),
        "action_dim": int(actions.shape[1]),
        "video_info": video_info,
    }

    with open(os.path.join(episode_out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] steps={usable_steps} state_dim={state.shape[1]} action_dim={actions.shape[1]}")
    return {
        "episode_out_dir": episode_out_dir,
        "meta": meta,
        "manifest": manifest,
    }


def main():
    episode_dirs = sorted(
        d for d in glob.glob(os.path.join(TASK_ROOT, "*")) if os.path.isdir(d)
    )[:3]

    print("found episode dirs:", len(episode_dirs))
    if not episode_dirs:
        print("No episode dirs found.")
        return

    all_rows = []
    processed = 0

    for i, ep_dir in enumerate(episode_dirs):
        result = process_episode(ep_dir, processed)
        if result is None:
            continue

        episode_out_dir = result["episode_out_dir"]
        state = np.load(os.path.join(episode_out_dir, "state.npy"))
        actions = np.load(os.path.join(episode_out_dir, "actions_23.npy"))

        for rec in result["manifest"]:
            t = rec["timestep"]
            row = {
                "episode_id": rec["episode_local_id"],
                "episode_name": rec["episode_name"],
                "timestep": t,
                "image_path": rec["image_path"],
                "instruction": rec["instruction"],
                "state": state[t].tolist(),
                "action": actions[t].tolist(),
            }
            all_rows.append(row)

        processed += 1

    jsonl_path = os.path.join(OUT_ROOT, "dataset.jsonl")
    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "num_processed_episodes": processed,
        "num_total_rows": len(all_rows),
        "task_root": TASK_ROOT,
        "output_root": OUT_ROOT,
    }

    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== done ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
