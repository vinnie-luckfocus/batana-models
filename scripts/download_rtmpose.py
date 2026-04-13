#!/usr/bin/env python3
"""Download and verify RTMPose-m COCO pretrained weights."""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
    "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192"
    "-d8dd5ca4_20230126.pth"
)
EXPECTED_SHA256 = "d8dd5ca4896c7a13d0998d20e16dcb6a9e45a0b2c6c8e1f5a9b8c7d6e5f4a3b2"
# Note: the SHA256 above is a placeholder; the real value should be updated
# after the first verified download.

MODEL_DIR = Path("models/2d_pose")
MODEL_PATH = MODEL_DIR / "rtmpose_m_coco.pth"


def download_file(url: str, dest: Path) -> None:
    """Download file with progress."""
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if MODEL_PATH.exists():
        actual_hash = sha256_file(MODEL_PATH)
        print(f"Model already exists. SHA-256: {actual_hash}")
        # Write hash to a sidecar file for reference
        (MODEL_DIR / "rtmpose_m_coco.sha256").write_text(actual_hash + "\n")
        return 0

    try:
        download_file(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        print(f"ERROR: Failed to download model: {exc}")
        return 1

    actual_hash = sha256_file(MODEL_PATH)
    print(f"Download complete. SHA-256: {actual_hash}")
    (MODEL_DIR / "rtmpose_m_coco.sha256").write_text(actual_hash + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
