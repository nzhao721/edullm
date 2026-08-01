"""Colab cell: load FineWeb SmolLM2 tokenized data from Google Drive or rclone."""

# --- Option A: Drive mount (Colab account has access to the folder) ---
# from google.colab import drive
# drive.mount("/content/drive")
# DATA = "/content/drive/MyDrive/edullm/fineweb-edu-1b-smollm2-tokenized"

# --- Option B: rclone (upload account != Colab login) ---
# !apt-get -qq install rclone
# !rclone config create gdrive-colab drive scope drive config_is_local false
# !rclone config reconnect gdrive-colab:
# !rclone copy gdrive-colab:edullm/fineweb-edu-1b-smollm2-tokenized /content/data --progress

DATA = "/content/data"  # or Drive path above

from pathlib import Path
import json
import numpy as np

data_dir = Path(DATA)
meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
tokens_path = data_dir / "train_tokens.bin"
num_tokens = int(meta["num_tokens"])
tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r", shape=(num_tokens,))
print(f"Loaded {num_tokens:,} tokens from {tokens_path}")
print(f"HF source: {meta.get('hf_path')}; tokenizer: {meta.get('tokenizer')}")
