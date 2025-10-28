import cv2
import numpy as np
import os
import glob
import argparse

def ai_like_enhance(img):
    # Step 1: Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Step 2: LAB Contrast (like Google auto)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Step 3: Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.5
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Step 4: Gamma for vibrance
    gamma = 1.1
    sharpened = np.power(sharpened / 255.0, gamma) * 255.0
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    for filepath in files:
        img = cv2.imread(filepath)
        if img is None: continue
        enhanced = ai_like_enhance(img)
        basename = os.path.basename(filepath)
        outpath = os.path.join(output_dir, f"enhanced_{basename}")
        cv2.imwrite(outpath, enhanced)
        print(f"✅ Processed: {basename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch AI-like Image Enhance")
    parser.add_argument('--input', '-i', required=True, help='Input folder')
    parser.add_argument('--output', '-o', default='enhanced_output', help='Output folder')
    args = parser.parse_args()
    batch_process(args.input, args.output)
    print("🎉 All done!")