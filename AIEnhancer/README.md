# Photo Batch Enhancer (Open Models)

Batch process large photo sets using open-source restoration/enhancement models:
- **Real-ESRGAN** (upscaling / general restoration)
- **GFPGAN** (face restoration)
- **CodeFormer** (face restoration with controllable fidelity)
- **Auto tone** (neutral white balance + CLAHE local contrast)
- (Optional) **HDRNet-like** tone mapping proxy via fast local Laplacian/CLAHE — no heavy weights needed.

> ⚠️ You must provide the model weights (download once) for Real-ESRGAN / GFPGAN / CodeFormer.
>
> This script is *modular*: if a model or weight is missing, that stage is skipped with a warning.

## Install

```bash
# (Recommended) New virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Get Weights (one-time)
- **Real-ESRGAN**: `RealESRGAN_x2plus.pth` or `RealESRGAN_x4plus.pth` (official repo releases)
- **GFPGAN**: `GFPGANv1.4.pth`
- **CodeFormer**: `CodeFormer.pth` (release weights)
Place them under `./weights/` (or pass `--realesrgan-weights`, `--gfpgan-weights`, `--codeformer-weights`).

## Run

```bash
python batch_enhance.py   --in ./photos_in   --out ./photos_out   --max-size 24   --autotone   --realesrgan 2   --gfpgan 1.0   --codeformer 0.7   --jobs 4
```

- `--autotone` enables white balance + local contrast (fast).
- `--realesrgan {2|4}` sets upscale factor (and enables Real-ESRGAN stage).
- `--gfpgan 1.0` enables GFPGAN (the number is "strength" ∈ [0,1], 1.0 = full face restore).
- `--codeformer 0.7` enables CodeFormer with fidelity 0.7 (0 = more restor., 1 = more fidelity).
- Use both `--gfpgan` and `--codeformer` if you want a *try-both pick-best* strategy (script compares sharpness/LPIPS-like metric and chooses the better).

## Tips
- For *non-face* landscapes, you likely just want `--autotone` + `--realesrgan 2`.
- For portraits, add `--gfpgan` or `--codeformer`.
- Set `--jobs` to number of CPU threads; GPU use is automatic if CUDA is available.
- EXIF and ICC are preserved when possible (via `pyvips`).

## Google Photos I/O (optional)
Use this script locally. To pull/push from Google Photos, wrap this with a separate fetch/upload tool using the Google Photos Library API, then point `--in` and `--out` to the local sync folders.
