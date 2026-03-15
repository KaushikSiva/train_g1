from huggingface_hub import snapshot_download

repo_id = "LightwheelAI/lightwheel_tasks"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir="data",
    allow_patterns=[
        "lightwheel_robocasa_tasks_g1_wbc/BeverageOrganization/*/trajectories.hdf5",
        "lightwheel_robocasa_tasks_g1_wbc/BeverageOrganization/*/isaac_replay_state_product.mp4",
    ],
    local_dir_use_symlinks=False,
)

print("done")
