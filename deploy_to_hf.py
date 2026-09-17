import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

DEPLOYMENT_DIR = Path(__file__).resolve().parent / "deployment"

def deploy(token: str, repo_name: str = "sentiment-analysis-ai"):
    """Deploy the self-contained deployment folder to Hugging Face Spaces."""
    token = token.strip().strip("'").strip('"')
    api = HfApi(token=token)
    
    print("🔑 Authenticating with Hugging Face...")
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✅ Authenticated as Hugging Face user: '{username}'")
    except Exception as exc:
        print(f"❌ Authentication Failed: {exc}")
        print("\nPlease ensure your token is a valid Write Token from https://huggingface.co/settings/tokens")
        sys.exit(1)
        
    repo_id = f"{username}/{repo_name}"
    
    print(f"📦 Preparing Hugging Face Space repository: {repo_id}...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="streamlit",
            private=False,
            token=token,
            exist_ok=True
        )
        print(f"✅ Space repository ready: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"⚠️ Note on repository initialization: {e}")

    print(f"🚀 Uploading application files from {DEPLOYMENT_DIR} to Hugging Face Space...")
    try:
        api.upload_folder(
            folder_path=str(DEPLOYMENT_DIR),
            repo_id=repo_id,
            repo_type="space",
            token=token,
            ignore_patterns=[".git/**", "__pycache__/**", "*.pyc"]
        )
    except Exception as exc:
        print(f"❌ Upload Failed: {exc}")
        print("If permissions failed, ensure your token has WRITE access at https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    public_url = f"https://huggingface.co/spaces/{repo_id}"
    direct_url = f"https://{username}-{repo_name.replace('_', '-')}.hf.space"
    
    print("\n" + "=" * 60)
    print("🎉 DEPLOYMENT SUCCESSFUL!")
    print(f"🌐 Public Space URL : {public_url}")
    print(f"⚡ Direct Web App URL: {direct_url}")
    print("=" * 60 + "\n")
    return public_url, direct_url

if __name__ == "__main__":
    token = os.getenv("HF_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
        
    if not token:
        print("❌ Error: No Hugging Face token provided.")
        print("\nUsage:")
        print("  python deploy_to_hf.py <YOUR_HF_WRITE_TOKEN>")
        print("\nGet your token here: https://huggingface.co/settings/tokens")
        sys.exit(1)
        
    deploy(token)

