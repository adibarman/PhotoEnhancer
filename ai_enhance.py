#CG
#!/usr/bin/env python3
"""
ai_enhance.py — a lightweight, Google-Photos-style "AI Enhance" pipeline.

Features (auto, content-aware-ish):
- White balance (gray-world with highlights protection)
- Exposure & local contrast (CLAHE on L* channel + adaptive gamma)
- De-noise (fastNlMeans) with scene-aware strength
- Sharpen (unsharp masking) with halo limiting
- Vibrance (boost low-sat colors more than already saturated ones)
- Optional portrait-friendly skin protection
- Simple scene heuristics (low-light, high-noise, flat contrast) to tune parameters
- NEW presets: landscape, filmic, product (in addition to natural, vivid, portrait, lowlight)
- Optional filmic midtone S-curve

Batch processing & CLI included.

Requirements:
  pip install opencv-python numpy

Usage (CLI):
  python ai_enhance.py input.jpg                    # writes input_enhanced.jpg next to it
  python ai_enhance.py /path/to/folder              # enhances all common image files
  python ai_enhance.py input.jpg -o out.jpg         # custom output
  python ai_enhance.py input.jpg --preset vivid     # presets: natural, vivid, portrait, lowlight, landscape, filmic, product
  python ai_enhance.py input.jpg --strength 0.7     # 0.0 .. 1.0 master strength
  python ai_enhance.py input.jpg --no-skin-protect  # disable skin protection

Notes:
- This is not a ML model; it's a carefully-tuned classical pipeline that mimics typical "auto enhance" behavior.
- For RAW files, convert to 16-bit TIFF/JPG via your RAW converter first, then run this script.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import cv2
import numpy as np

# ---------------------------- Utilities ----------------------------

def _to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)

def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)

def _estimate_noise_gray(img_gray: np.ndarray) -> float:
    lap = cv2.Laplacian(img_gray, cv2.CV_32F)
    var = float(lap.var())
    return var

def _image_stats(img: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = float(np.mean(hsv[...,1]))  # 0..1
    noise_proxy = _estimate_noise_gray(gray)
    return dict(brightness=brightness, contrast=contrast, saturation=sat, noise=noise_proxy)

# ---------------------- White Balance (gray-world) ----------------------

def white_balance_gray_world(img: np.ndarray, highlight_protect=0.05) -> np.ndarray:
    percent = np.clip(highlight_protect, 0.0, 0.45)
    flat = img.reshape(-1, 3)
    mask = np.all(flat < (1.0 - percent), axis=1)
    if np.count_nonzero(mask) < 100:
        mask = slice(None)
    means = flat[mask].mean(axis=0) + 1e-6
    scale = means.mean() / means
    balanced = np.clip(img * scale, 0, 1)
    return balanced

# ---------------------- Exposure & Local Contrast ----------------------

def adaptive_gamma(img: np.ndarray, target=0.5, mix=1.0) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean = float(np.mean(gray)) + 1e-6
    gamma = np.log(max(target, 1e-6)) / np.log(max(mean, 1e-6))
    gamma = np.clip(gamma, 0.5, 2.0)
    if mix < 1.0:
        gamma = 1.0 * (1.0 - mix) + gamma * mix
    inv = 1.0 / gamma
    out = np.power(np.clip(img, 0, 1), inv)
    return out

def clahe_l_channel(img: np.ndarray, clip=2.0, tile=(8,8)) -> np.ndarray:
    lab = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB).astype(np.uint8)
    return _to_float(out)

# ---------------------- Denoise & Sharpen ----------------------

def denoise_nlmeans(img: np.ndarray, strength=0.3) -> np.ndarray:
    if strength <= 0:
        return img
    bgr = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2BGR)
    h = int(10 + 30 * strength)  # 10..40
    den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
    rgb = cv2.cvtColor(den, cv2.COLOR_BGR2RGB)
    return _to_float(rgb)

def unsharp_mask(img: np.ndarray, radius=1.5, amount=1.0, threshold=0.0) -> np.ndarray:
    k = max(1, int(radius*3)//2*2+1)
    blurred = cv2.GaussianBlur(img, (k,k), radius)
    mask = img - blurred
    mask = np.tanh(mask * 2.0)
    if threshold > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        contrast = cv2.Laplacian(gray, cv2.CV_32F)
        thr = (np.abs(contrast) > threshold).astype(np.float32)[...,None]
        mask = mask * thr
    out = img + amount * mask
    return np.clip(out, 0, 1)

# ---------------------- Vibrance (smart saturation) ----------------------

def vibrance(img: np.ndarray, gain=0.2) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    boost = (1.0 - s) * gain * 1.5 + gain * 0.2
    s2 = np.clip(s + boost * (1.0 - s), 0, 1)
    hsv2 = cv2.merge([h, s2, v])
    out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)
    return out

# ---------------------- Filmic S-curve ----------------------

def filmic_scurve(img: np.ndarray, strength=0.35) -> np.ndarray:
    """Gentle midtone S-curve; compresses extremes, boosts midtone contrast."""
    t = np.clip(img, 0, 1)
    mid = t - 0.5
    sc = mid * (1.0 - np.abs(2.0*mid))  # emphasize midtones
    y = t + strength * sc
    return np.clip(y, 0, 1)

# ---------------------- Skin Protection ----------------------

def skin_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    mask = (
        (S > 0.15) & (S < 0.8) & 
        (V > 0.2) & (V < 0.95) &
        (
            ((H >= 0.0) & (H <= 0.12)) |
            ((H >= 0.92) & (H <= 1.0))
        )
    )
    mask = mask.astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    return (mask.astype(np.float32)/255.0)[...,None]

def blend_protect(base: np.ndarray, altered: np.ndarray, mask: np.ndarray, keep=0.6) -> np.ndarray:
    return altered * (1 - mask*keep) + base * (mask*keep)

# ---------------------- Presets & Heuristics ----------------------

@dataclass
class EnhanceParams:
    wb_highlight_protect: float = 0.05
    gamma_target: float = 0.5
    gamma_mix: float = 1.0
    clahe_clip: float = 2.0
    clahe_tile: Tuple[int,int] = (8,8)
    denoise_strength: float = 0.25
    sharpen_radius: float = 1.5
    sharpen_amount: float = 0.6
    sharpen_threshold: float = 0.0
    vibrance_gain: float = 0.25
    skin_protect_keep: float = 0.5
    use_skin_protect: bool = True
    # filmic tone
    use_filmic: bool = False
    filmic_strength: float = 0.0

PRESETS = {
    "natural": EnhanceParams(),
    "vivid": EnhanceParams(vibrance_gain=0.4, clahe_clip=3.0, sharpen_amount=0.9, denoise_strength=0.2),
    "portrait": EnhanceParams(denoise_strength=0.3, sharpen_amount=0.45, vibrance_gain=0.2, use_skin_protect=True, skin_protect_keep=0.7),
    "lowlight": EnhanceParams(gamma_target=0.6, denoise_strength=0.45, clahe_clip=3.0, sharpen_amount=0.5, vibrance_gain=0.3),
    # NEW
    "landscape": EnhanceParams(
        wb_highlight_protect=0.06,
        gamma_target=0.5,
        clahe_clip=3.0,
        clahe_tile=(8,8),
        denoise_strength=0.15,
        sharpen_radius=1.8,
        sharpen_amount=1.0,
        vibrance_gain=0.45,
        use_skin_protect=False
    ),
    "filmic": EnhanceParams(
        wb_highlight_protect=0.05,
        gamma_target=0.48,
        clahe_clip=2.0,
        denoise_strength=0.2,
        sharpen_radius=1.4,
        sharpen_amount=0.5,
        vibrance_gain=0.18,
        use_skin_protect=False,
        use_filmic=True,
        filmic_strength=0.35
    ),
    "product": EnhanceParams(
        wb_highlight_protect=0.08,
        gamma_target=0.5,
        clahe_clip=2.5,
        denoise_strength=0.1,
        sharpen_radius=1.2,
        sharpen_amount=1.1,
        vibrance_gain=0.12,
        use_skin_protect=False
    )
}

def tune_by_scene(img: np.ndarray, p: EnhanceParams) -> EnhanceParams:
    stats = _image_stats(img)
    brightness, contrast, sat, noise = stats["brightness"], stats["contrast"], stats["saturation"], stats["noise"]
    p = EnhanceParams(**vars(p))  # copy
    if brightness < 0.35:
        p.gamma_target = max(p.gamma_target, 0.6)
        p.denoise_strength = max(p.denoise_strength, 0.4)
        p.sharpen_amount = min(p.sharpen_amount, 0.55)
    if contrast < 0.08:
        p.clahe_clip = max(p.clahe_clip, 3.0)
    if sat > 0.45:
        p.vibrance_gain = min(p.vibrance_gain, 0.18)
    if noise > 15.0:
        p.sharpen_amount = min(p.sharpen_amount, 0.55)
    return p

# ---------------------- Main Enhance ----------------------

def enhance_image(img_bgr: np.ndarray, preset="natural", strength=1.0, no_skin_protect=False) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    f = _to_float(img)

    params = PRESETS.get(preset, PRESETS["natural"])
    params = tune_by_scene(f, params)

    s = np.clip(float(strength), 0.0, 1.0)
    wb_hp = np.interp(s, [0,1], [0.02, params.wb_highlight_protect])
    gamma_mix = np.interp(s, [0,1], [0.2, params.gamma_mix])
    clahe_clip = np.interp(s, [0,1], [1.2, params.clahe_clip])
    denoise_s = np.interp(s, [0,1], [0.0, params.denoise_strength])
    sharp_amt = np.interp(s, [0,1], [0.20, params.sharpen_amount])
    vibr_gain = np.interp(s, [0,1], [0.10, params.vibrance_gain])
    skin_keep = np.interp(s, [0,1], [0.3, params.skin_protect_keep])
    filmic_s = np.interp(s, [0,1], [0.15, params.filmic_strength]) if params.use_filmic else 0.0

    base = f
    f = white_balance_gray_world(f, highlight_protect=wb_hp)
    f = adaptive_gamma(f, target=params.gamma_target, mix=gamma_mix)
    f = clahe_l_channel(f, clip=clahe_clip, tile=params.clahe_tile)
    f = denoise_nlmeans(f, strength=denoise_s)
    sharp = unsharp_mask(f, radius=params.sharpen_radius, amount=sharp_amt, threshold=params.sharpen_threshold)
    f = np.clip(sharp, 0, 1)
    if params.use_filmic and filmic_s > 0:
        f = filmic_scurve(f, strength=filmic_s)
    f = vibrance(f, gain=vibr_gain)

    if params.use_skin_protect and not no_skin_protect:
        mask = skin_mask(base)
        f = blend_protect(f, base, mask, keep=skin_keep)

    out = _to_uint8(f)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ---------------------- I/O Helpers ----------------------

VALID_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def output_path_for(input_path: str, output: str|None) -> str:
    if output:
        return output
    root, ext = os.path.splitext(input_path)
    return f"{root}_enhanced{ext or '.jpg'}"

def process_file(path: str, out_path: str|None, preset="natural", strength=1.0, no_skin_protect=False, overwrite=False) -> str:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    enhanced = enhance_image(img, preset=preset, strength=strength, no_skin_protect=no_skin_protect)
    dest = output_path_for(path, out_path)
    if (not overwrite) and os.path.exists(dest):
        root, ext = os.path.splitext(dest)
        i = 1
        cand = f"{root}-{i}{ext}"
        while os.path.exists(cand):
            i += 1
            cand = f"{root}-{i}{ext}"
        dest = cand
    ext = os.path.splitext(dest)[1].lower() or ".jpg"
    encode_ext = ".jpg" if ext.lower() not in VALID_EXTS else ext.lower()
    ok, buf = cv2.imencode(encode_ext, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode output.")
    buf.tofile(dest)
    return dest

def process_folder(folder: str, preset="natural", strength=1.0, no_skin_protect=False, overwrite=False) -> None:
    from glob import glob
    paths = []
    for ext in VALID_EXTS:
        paths.extend(glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob(os.path.join(folder, f"*{ext.upper()}")))
    if not paths:
        print("No images found.")
        return
    for p in sorted(set(paths)):
        try:
            out = process_file(p, None, preset=preset, strength=strength, no_skin_protect=no_skin_protect, overwrite=overwrite)
            print(f"✔ {p} -> {out}")
        except Exception as e:
            print(f"✖ {p}: {e}")

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser(description='Google-Photos-like "AI Enhance" pipeline')
    ap.add_argument('input', help='Path to an image file or a folder of images')
    ap.add_argument('-o', '--output', help='Output file path (for single image input)')
    ap.add_argument('--preset', choices=list(PRESETS.keys()), default='natural',
                    help='Enhancement preset: ' + ', '.join(PRESETS.keys()))
    ap.add_argument('--strength', type=float, default=1.0, help='Overall effect strength (0..1)')
    ap.add_argument('--no-skin-protect', action='store_true', help='Disable portrait/skin protection')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite if output exists')
    args = ap.parse_args()

    path = args.input
    if os.path.isdir(path):
        process_folder(path, preset=args.preset, strength=args.strength, no_skin_protect=args.no_skin_protect, overwrite=args.overwrite)
    else:
        out = process_file(path, args.output, preset=args.preset, strength=args.strength, no_skin_protect=args.no_skin_protect, overwrite=args.overwrite)
        print(out)

if __name__ == '__main__':
    main()
