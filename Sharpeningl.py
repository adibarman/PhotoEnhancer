#GG
from skimage import io, filters, util
import numpy as np
import matplotlib.pyplot as plt

def sharpen_image_unsharp_mask(image_path, radius=1.0, amount=1.0):
    """
    Applies Unsharp Masking to sharpen an image.
    
    Args:
        image_path (str): Path to the input image.
        radius (float): Standard deviation of the Gaussian blur used for the mask. 
                        Smaller radius sharpens fine details, larger sharpens larger features.
        amount (float): The strength of the sharpening effect. Higher amount means more sharpness.
    """
    # 1. Load the image in floating point format for processing
    img = util.img_as_float(io.imread(image_path))
    
    # 2. Apply Gaussian blur to create the "unsharp" mask
    blurred = filters.gaussian(img, sigma=radius, channel_axis=-1)
    
    # 3. Calculate the mask (original - blurred)
    mask = img - blurred
    
    # 4. Sharpen the image (original + amount * mask)
    sharpened_img = img + mask * amount
    
    # 5. Clip values to the valid range [0, 1] and convert back to uint8
    sharpened_img = np.clip(sharpened_img, 0, 1)
    sharpened_img_uint8 = util.img_as_ubyte(sharpened_img)
    
    # 6. Save the result
    output_path = image_path.replace(".jpg", "_sharpened.jpg").replace(".png", "_sharpened.png")
    io.imsave(output_path, sharpened_img_uint8)
    print(f"Sharpened image saved to {output_path}")

# --- Example Usage ---
# sharpen_image_unsharp_mask('your_image.jpg', radius=1.5, amount=2.0)