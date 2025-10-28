#PP
import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from glob import glob
# If using Real-ESRGAN, GFPGAN, or similar deep-learning models, ensure to install their libraries and dependencies.

def enhance_image(img_path, output_path):
    # Load image with OpenCV
    img = cv2.imread(img_path)
    # Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7)
    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    # Save intermediate OpenCV output
    temp_path = output_path.replace('.jpg', '_cv.jpg')
    cv2.imwrite(temp_path, img)

    # Further enhancements with PIL
    pil_img = Image.open(temp_path)
    # Increase contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.25)
    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.3)
    # Increase color
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(1.15)
    pil_img.save(output_path)
    # Remove temp file
    os.remove(temp_path)

def batch_enhance(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = glob(os.path.join(input_folder, '*.jpg')) + glob(os.path.join(input_folder, '*.png'))
    for img_path in image_files:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        enhance_image(img_path, output_path)
        print(f'Processed: {filename}')

# Example usage:
batch_enhance('input_images', 'enhanced_images')
