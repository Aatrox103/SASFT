import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
from huggingface_hub import login
# Set your own access token if needed
# access_token_read = 'YOUR_OWN_ACCESS_TOKEN'
# login(token = access_token_read)
download_path = "./Llama3_1-8B-Base-LXR-8x"
snapshot_download(repo_id="fnlp/Llama3_1-8B-Base-LXR-8x", local_dir=download_path)

download_path = "./gemma-scope-2b-pt-res"
snapshot_download(repo_id="google/gemma-scope-2b-pt-res", local_dir=download_path)

download_path = "./gemma-scope-9b-pt-res"
snapshot_download(repo_id="google/gemma-scope-9b-pt-res", local_dir=download_path)
