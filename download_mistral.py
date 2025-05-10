#!/usr/bin/env python3
import os
import json
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
LOCAL_DIR = "models/Mistral-7B-Instruct-v0.2"
TOKEN     = os.environ.get("HF_TOKEN")  # export HF_TOKEN first if gated

os.makedirs(LOCAL_DIR, exist_ok=True)

# 1) Download index files
for fname in [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]:
    print(f"Downloading {fname}…")
    hf_hub_download(
        repo_id=MODEL_ID,
        filename=fname,
        local_dir=LOCAL_DIR,
        use_auth_token=TOKEN,
        local_files_only=False
    )

# 2) Read safetensors index and schedule each shard
with open(os.path.join(LOCAL_DIR, "model.safetensors.index.json")) as f:
    st_index = json.load(f)
safetensor_shards = list(set(st_index["weight_map"].values()))

# 3) Read bin index and schedule each shard (optional; skip if you only want safetensors)
with open(os.path.join(LOCAL_DIR, "pytorch_model.bin.index.json")) as f:
    bin_index = json.load(f)
bin_shards = list(set(bin_index["weight_map"].values()))

# Combine them or pick one format:
# To download only safetensors, use safetensor_shards.
# To download only bin, use bin_shards.
shards = safetensor_shards  # or bin_shards

# 4) Download shards in parallel
def download_shard(shard_name):
    print(f"Downloading shard {shard_name}…")
    hf_hub_download(
        repo_id=MODEL_ID,
        filename=shard_name,
        local_dir=LOCAL_DIR,
        use_auth_token=TOKEN,
        local_files_only=False
    )
    return shard_name

with ThreadPoolExecutor(max_workers=6) as exe:
    futures = [exe.submit(download_shard, s) for s in shards]
    for fut in as_completed(futures):
        print("Finished:", fut.result())

print("All shards downloaded!")
