import h5py

path = "data/lightwheel_robocasa_tasks_g1_wbc/BeverageOrganization/BeverageOrganization_1763692193809507/trajectories.hdf5"

keywords = [
    "image", "rgb", "camera", "frame", "video",
    "obs/actions", "processed_actions", "joint_targets",
    "states/articulation/robot", "ee_pose"
]

with h5py.File(path, "r") as f:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            if any(k in name for k in keywords):
                print(f"DATASET: {name} shape={obj.shape} dtype={obj.dtype}")
        else:
            if any(k in name for k in keywords):
                print(f"GROUP:   {name}")
    f.visititems(walk)
