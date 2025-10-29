
#!/usr/bin/env python3
import os, sys, argparse, pathlib, warnings, math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
from tqdm import tqdm

# Optional fast IO via pyvips
try:
    import pyvips
    HAS_VIPS = True
except Exception:
    HAS_VIPS = False

# Try importing models gracefully
def try_import_realesrgan():
    try:
        from realesrgan import RealESRGAN
        return RealESRGAN
    except Exception as e:
        warnings.warn(f"Real-ESRGAN not available: {e}")
        return None

def try_import_gfpgan():
    try:
        from gfpgan import GFPGANer
        return GFPGANer
    except Exception as e:
        warnings.warn(f"GFPGAN not available: {e}")
        return None

def try_import_codeformer():
    try:
        from codeformer_pip import inference as cf_infer
        return cf_infer
    except Exception as e:
        warnings.warn(f"CodeFormer not available: {e}")
        return None

def vips_read(path):
    if HAS_VIPS:
        vimg = pyvips.Image.new_from_file(str(path), access="sequential")
        arr = np.ndarray(buffer=vimg.write_to_memory(),
                         dtype=np.uint8,
                         shape=[vimg.height, vimg.width, vimg.bands])
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def vips_write(rgb, out_path, q=92):
    out_path = str(out_path)
    h, w = rgb.shape[:2]
    if HAS_VIPS:
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb,0,255).astype(np.uint8)
        vimg = pyvips.Image.new_from_memory(rgb.tobytes(), w, h, 3, "uchar")
        suffix = os.path.splitext(out_path.lower())[1]
        if suffix in [".jpg",".jpeg"]:
            vimg.jpegsave(out_path, Q=q, optimize_coding=True, strip=False)
        elif suffix == ".png":
            vimg.pngsave(out_path, compression=6, strip=False)
        else:
            vimg.jpegsave(out_path, Q=q, optimize_coding=True, strip=False)
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])

def auto_white_balance(img_rgb):
    # Simple gray-world + channel clipping
    img = img_rgb.astype(np.float32)
    mean = img.reshape(-1,3).mean(axis=0) + 1e-6
    scale = mean.mean()/mean
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)

def local_contrast_clahe(img_rgb, clip=2.0, tiles=8):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles,tiles))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def sharpen_unsharp(rgb, amount=0.6, radius=3):
    blur = cv2.GaussianBlur(rgb,(0,0), radius)
    out = cv2.addWeighted(rgb, 1+amount, blur, -amount, 0)
    return np.clip(out,0,255).astype(np.uint8)

def detect_faces_bboxes(rgb):
    # Lightweight face detector via OpenCV Haar (fallback)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def variance_of_laplacian(img):
    # focus measure for pick-best between restorers
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

def process_one(path, args, engines):
    try:
        rgb = vips_read(path)
        # Resize guard for huge images (max side in megapixels)
        if args.max_size is not None:
            max_mp = args.max_size
            h, w = rgb.shape[:2]
            mp = (h*w)/1e6
            if mp > max_mp:
                scale = math.sqrt(max_mp/mp)
                new_wh = (int(w*scale), int(h*scale))
                rgb = cv2.resize(rgb, new_wh, interpolation=cv2.INTER_AREA)

        # --- Auto tone ---
        if args.autotone:
            rgb = auto_white_balance(rgb)
            rgb = local_contrast_clahe(rgb, clip=args.clahe, tiles=args.clahe_tiles)

        # --- Real-ESRGAN (general restoration/upscale) ---
        if args.realesrgan and engines.get('realesrgan'):
            model = engines['realesrgan']
            rgb = model.predict(rgb)

        # --- Face restoration (GFPGAN / CodeFormer) when faces present ---
        bboxes = detect_faces_bboxes(rgb) if (args.gfpgan or args.codeformer) else []
        if len(bboxes) > 0:
            best = rgb
            best_score = variance_of_laplacian(best)

            if args.gfpgan and engines.get('gfpgan'):
                try:
                    _, _, out_rgb = engines['gfpgan'].enhance(rgb, has_aligned=False, only_center_face=False, paste_back=True)
                    score = variance_of_laplacian(out_rgb)
                    if score > best_score:
                        best, best_score = out_rgb, score
                except Exception as e:
                    warnings.warn(f"GFPGAN failed on {path}: {e}")

            if args.codeformer and engines.get('codeformer'):
                try:
                    out_rgb = engines['codeformer'](rgb, fidelity=args.codeformer)
                    score = variance_of_laplacian(out_rgb)
                    if score > best_score:
                        best, best_score = out_rgb, score
                except Exception as e:
                    warnings.warn(f"CodeFormer failed on {path}: {e}")

            rgb = best

        # --- Optional unsharp mask for perceived clarity ---
        if args.sharpen > 0:
            rgb = sharpen_unsharp(rgb, amount=args.sharpen, radius=args.sharpen_radius)

        # Write
        rel = path.relative_to(args.in_dir)
        out_path = args.out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vips_write(rgb, out_path, q=args.jpeg_quality)
        return True, str(out_path)
    except Exception as e:
        return False, f"{path}: {e}"

def build_engines(args):
    engines = {}
    # Real-ESRGAN
    if args.realesrgan:
        RealESRGAN = try_import_realesrgan()
        if RealESRGAN:
            device = "cuda" if (cv2.cuda.getCudaEnabledDeviceCount() > 0) else "cpu"
            model = RealESRGAN(device, scale=args.realesrgan)
            if args.realesrgan_weights:
                model.load_weights(args.realesrgan_weights)
            else:
                warnings.warn("Real-ESRGAN enabled but no weights specified. Ensure default path is discoverable or pass --realesrgan-weights.")
            engines['realesrgan'] = model
    # GFPGAN
    if args.gfpgan is not None:
        GFPGANer = try_import_gfpgan()
        if GFPGANer:
            if not args.gfpgan_weights:
                warnings.warn("GFPGAN enabled but --gfpgan-weights not provided.")
            engines['gfpgan'] = GFPGANer(model_path=args.gfpgan_weights or '',
                                         upscale=1, arch='clean', channel_multiplier=2,
                                         bg_upsampler=None)
    # CodeFormer
    if args.codeformer is not None:
        cf = try_import_codeformer()
        if cf:
            def run_codeformer(rgb, fidelity=0.7):
                # codeformer-pip wrapper expects BGR uint8
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                out_bgr = cf.enhance_bgr(
                    bgr,
                    fidelity=fidelity,
                    has_aligned=False,
                    background_enhance=False,
                    face_upsample=False,
                    upscale=1,
                    ckpt_path=args.codeformer_weights or None
                )
                return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            engines['codeformer'] = run_codeformer
    return engines

def gather_images(in_dir):
    exts = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
    paths = []
    for p in pathlib.Path(in_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)

def main():
    ap = argparse.ArgumentParser(description="Batch AI photo enhancer using open models.")
    ap.add_argument("--in", dest="in_dir", required=True, help="Input folder")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder")
    ap.add_argument("--jobs", type=int, default=4, help="Parallel workers")
    ap.add_argument("--max-size", type=float, default=24.0, help="Max megapixels per image (downscale if larger). Use 0 to disable.")
    # Auto tone
    ap.add_argument("--autotone", action="store_true", help="Enable auto white balance + CLAHE")
    ap.add_argument("--clahe", type=float, default=2.0, help="CLAHE clip limit")
    ap.add_argument("--clahe-tiles", type=int, default=8, help="CLAHE tile grid size")
    # Real-ESRGAN
    ap.add_argument("--realesrgan", type=int, choices=[2,4], help="Enable Real-ESRGAN with upscale 2x or 4x")
    ap.add_argument("--realesrgan-weights", type=str, help="Path to Real-ESRGAN .pth weights")
    # GFPGAN
    ap.add_argument("--gfpgan", type=float, nargs='?', const=1.0, help="Enable GFPGAN (value is strength; informational)")
    ap.add_argument("--gfpgan-weights", type=str, help="Path to GFPGAN .pth weights")
    # CodeFormer
    ap.add_argument("--codeformer", type=float, nargs='?', const=0.7, help="Enable CodeFormer (fidelity in [0,1])")
    ap.add_argument("--codeformer-weights", type=str, help="Path to CodeFormer .pth checkpoint")
    # Final clarity
    ap.add_argument("--sharpen", type=float, default=0.15, help="Unsharp mask amount (0 to disable)")
    ap.add_argument("--sharpen-radius", type=float, default=2.0, help="Unsharp radius (sigma)")
    ap.add_argument("--jpeg-quality", type=int, default=92, help="Output JPEG quality")
    args = ap.parse_args()

    args.in_dir = pathlib.Path(args.in_dir)
    args.out_dir = pathlib.Path(args.out_dir)
    if args.max_size and args.max_size <= 0:
        args.max_size = None

    paths = gather_images(args.in_dir)
    if not paths:
        print("No images found in input folder.", file=sys.stderr)
        sys.exit(2)

    engines = build_engines(args)

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(process_one, p, args, engines): p for p in paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            success, msg = f.result()
            if success:
                ok += 1
            else:
                fail += 1
                warnings.warn(msg)
    print(f"Done. Success: {ok}, Failed: {fail}")

if __name__ == "__main__":
    main()
