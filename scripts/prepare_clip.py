"""Download the pinned research model into the project and retain provenance."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    os.environ["HF_HOME"] = str(ROOT / "local/models/.hf-eval")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    from huggingface_hub import snapshot_download

    protocol_hash = sha256(ROOT / "notes/week3_metrics.md")
    destination = ROOT / "local/models/clip-vit-base-patch32"
    snapshot_download(
        MODEL, revision=REVISION, local_dir=destination, max_workers=2,
        allow_patterns=["*.json", "*.txt", "README.md", "pytorch_model.bin"],
    )
    license_url = "https://raw.githubusercontent.com/openai/CLIP/main/LICENSE"
    license_text = urlopen(license_url, timeout=30).read()
    if b"MIT License" not in license_text or b"Copyright (c) 2021 OpenAI" not in license_text:
        raise RuntimeError("Unexpected license content")
    (destination / "LICENSE.openai-CLIP.txt").write_bytes(license_text)
    provenance = {
        "model": MODEL, "revision": REVISION, "license_url": license_url,
        "protocol_before_download_sha256": protocol_hash,
        "files": {p.name: sha256(p) for p in destination.iterdir() if p.is_file()},
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Pinned CLIP ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
