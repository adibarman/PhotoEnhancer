# mnaus
import cv2
import numpy as np

def auto_enhance(image):
    """
    Automatically enhances the brightness, contrast, and color of an image.
    """
    # --- Automatic Brightness and Contrast ---
    # Convert to LAB color space to separate lightness from color
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # --- Automatic Color Balance ---
    # A simple white-balance algorithm
    result = enhanced_image.copy()
    channels = cv2.split(result)
    avg_b = np.mean(channels[0])
    avg_g = np.mean(channels[1])
    avg_r = np.mean(channels[2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    channels[0] = np.clip(channels[0] * scale_b, 0, 255).astype(np.uint8)
    channels[1] = np.clip(channels[1] * scale_g, 0, 255).astype(np.uint8)
    channels[2] = np.clip(channels[2] * scale_r, 0, 255).astype(np.uint8)

    balanced_image = cv2.merge(channels)

    return balanced_image

# # Example Usage:
# image = cv2.imread('your_image.jpg')
# enhanced = auto_enhance(image)
# cv2.imshow('Enhanced Image', enhanced)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
