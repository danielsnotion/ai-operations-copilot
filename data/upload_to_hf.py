import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path
load_dotenv()


# ================= CONFIG ================= #
HF_TOKEN = os.getenv("HF_TOKEN")  # export HF_TOKEN=xxx
USERNAME = "daniel1028"      
REPO_NAME = "ai-operations-copilot-data"

LOCAL_DATASET_PATH = "data/ai-operations-copilot-data"


# ================= VALIDATION ================= #
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not set. Run: export HF_TOKEN=your_token")

if not Path(LOCAL_DATASET_PATH).exists():
    raise ValueError(f"❌ Folder not found: {LOCAL_DATASET_PATH}")


# ================= INIT ================= #
api = HfApi(token=HF_TOKEN)
repo_id = f"{USERNAME}/{REPO_NAME}"


# ================= CREATE REPO ================= #
print("🚀 Creating dataset repo (if not exists)...")

try:
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=HF_TOKEN
    )
    print(f"✅ Repo ready: {repo_id}")
except Exception as e:
    print(f"⚠️ Repo creation skipped: {e}")


# ================= UPLOAD ================= #
print("📦 Uploading dataset...")

upload_folder(
    folder_path=LOCAL_DATASET_PATH,
    repo_id=repo_id,
    repo_type="dataset",
    token=HF_TOKEN
)

print("✅ Upload completed successfully!")


# ================= FINAL LINK ================= #
print("\n🔗 Dataset available at:")
print(f"https://huggingface.co/datasets/{repo_id}")