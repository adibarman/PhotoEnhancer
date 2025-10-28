#X

# The closest free, open-source Python alternative is Real-ESRGAN (general restoration) combined with GFPGAN (AI face enhancement). It produces near-identical results on old/low-res photos and supports full batch processing of folders.


# /*
# git clone https://github.com/xinntao/Real-ESRGAN.git
# cd Real-ESRGAN
# pip install basicsr facexlib gfpgan
# pip install -r requirements.txt
# python setup.py develop
# */

# #Download models
# wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights/

#Batch Process a Folder (One Command!)
# python inference_realesrgan.py \
#   -n RealESRGAN_x4plus \  # Best general model
#   -i input_folder \       # Input folder
#   -o output_folder \      # Output folder (auto-created)
#   --face_enhance \        # AI face restore like Google
#   --suffix _ai_enhanced \ # Output filename suffix
#   -s 4                    # Upscale 4x (or 2 for faster)

#Wrapper
import os
import subprocess
import argparse

def run_batch(input_dir, output_dir, model='RealESRGAN_x4plus', scale=4, face_enhance=True):
    cmd = [
        'python', 'inference_realesrgan.py',
        '-n', model,
        '-i', input_dir,
        '-o', output_dir,
        '-s', str(scale),
        '--suffix', '_ai_enhanced'
    ]
    if face_enhance:
        cmd += ['--face_enhance']
    subprocess.run(cmd)
    print(f"✅ Batch complete! Check {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input folder')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    run_batch(args.input, args.output, scale=args.scale)