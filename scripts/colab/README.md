# FineWeb tokenized corpus → Google Drive → Colab

Upload a **separate** Google account (not Google Drive for Desktop) using rclone.

## What gets uploaded

Default bundle is the full train memmaps (~11 GB). Pass `-TokensOnly` for tokens + metadata only (~3.8 GB).

## One-time setup (laptop)

```powershell
# From repo root on Windows
powershell -ExecutionPolicy Bypass -File scripts/colab/setup_gdrive_colab_remote.ps1
```

A browser window opens. **Sign in with the Google account you want for Colab** (not the Drive for Desktop account).

## Upload

```powershell
powershell -ExecutionPolicy Bypass -File scripts/colab/upload_fineweb_to_gdrive.ps1
```

Remote path: `gdrive-colab:edullm/fineweb-edu-1b-smollm2-tokenized/`

## Colab

**Option A — Colab logged into the same Google account as the upload**

```python
from google.colab import drive
drive.mount("/content/drive")
DATA = "/content/drive/MyDrive/edullm/fineweb-edu-1b-smollm2-tokenized"
```

**Option B — Colab on a different Google account**

Share the Drive folder from the upload account to your Colab account (Viewer is enough), then mount Drive in Colab as above.

**Option C — rclone inside Colab** (same upload account, any Colab login)

See `scripts/colab/colab_load_fineweb.py`.

## Slice 750M / 500M on Colab

```python
import numpy as np
from pathlib import Path

def slice_prefix(src_dir: Path, dst_dir: Path, max_tokens: int):
    dst_dir.mkdir(parents=True, exist_ok=True)
    src = np.memmap(src_dir / "train_tokens.bin", dtype=np.uint32, mode="r")
    n = min(max_tokens, len(src))
    dst = np.memmap(dst_dir / "train_tokens.bin", dtype=np.uint32, mode="w+", shape=(n,))
    dst[:] = src[:n]
    dst.flush()

# slice_prefix(Path(DATA), Path("/content/data-750m"), 750_000_000)
```

Full local slice (all sidecars): `scripts/farmshare/slice_tokenized_subset.py`.
