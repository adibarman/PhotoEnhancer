#GG
import cv2
import numpy as np
from PIL import Image

def denoise_image_fast(image_path, h=10, h_color=10, templateWindowSize=7, searchWindowSize=21):
    """
    Applies Non-Local Means Denoising to a color image.
    
    Args:
        image_path (str): Path to the input image.
        h (int): Parameter regulating filter strength for luminance components. 
                 Higher h removes more noise but may blur edges.
        h_color (int): Parameter regulating filter strength for color components.
        templateWindowSize (int): Size of the pixel neighborhood used to compute weights. Should be odd.
        searchWindowSize (int): Size of the search window. Should be odd.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Apply the fast Non-Local Means Denoising algorithm
    # fastNlMeansDenoisingColored is for color images (better than fastNlMeansDenoising)
    denoised_img = cv2.fastNlMeansDenoisingColored(
        img, 
        None, 
        h, 
        h_color, 
        templateWindowSize, 
        searchWindowSize
    )
    
    # 3. Save the result
    output_path = image_path.replace(".jpg", "_denoised.jpg").replace(".png", "_denoised.png")
    cv2.imwrite(output_path, denoised_img)
    print(f"Denoised image saved to {output_path}")

# --- Example Usage ---
# denoise_image_fast('your_noisy_image.jpg')