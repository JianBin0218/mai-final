"""Download the pinned Qwen2-Audio checkpoint into MAI_MODEL_DIR (default: model/)."""
from huggingface_hub import snapshot_download

from baseline import MODEL_DIR

REPO = "Qwen/Qwen2-Audio-7B-Instruct"
REVISION = "0a095220c30b7b31434169c3086508ef3ea5bf0a"


if __name__ == "__main__":
    print(snapshot_download(REPO, revision=REVISION, local_dir=MODEL_DIR, max_workers=8))
