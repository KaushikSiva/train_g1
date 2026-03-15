from huggingface_hub import hf_hub_download

repo_id = "LightwheelAI/lightwheel_tasks"

path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="lightwheel_robocasa_tasks_g1_wbc/BeverageOrganization/BeverageOrganization_1763692193809507/isaac_replay_state_product.mp4",
    local_dir="data"
)

print("Downloaded to:", path)