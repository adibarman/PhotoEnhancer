#GG
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the model path (ESRGAN for 4x Super Resolution)
# This is a pre-trained model that can increase image resolution by 4 times
MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def preprocess_image(image_path):
    """Loads image, converts it to a Tensor, and removes alpha channel if present."""
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # Remove alpha channel if present (model only supports 3 channels)
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_image = tf.cast(hr_image, tf.float32)
    # The model expects a batch of images, so we add a batch dimension
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    """Saves the output Tensor as a JPEG image."""
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        # Convert the Tensor back to a NumPy array of uint8 before saving
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(f"{filename}.jpg")
    print(f"Saved enhanced image as {filename}.jpg")

def enhance_image_esrgan(image_path, output_filename="enhanced_esrgan"):
    """
    Applies the ESRGAN model to enhance the image resolution.
    """
    print("Loading image and model...")
    hr_image = preprocess_image(image_path)
    
    # Load the pre-trained ESRGAN model
    model = hub.load(MODEL_PATH)

    print("Enhancing image...")
    # Run the image through the super-resolution model
    fake_image = model(hr_image)
    # Remove the batch dimension
    fake_image = tf.squeeze(fake_image)
    
    # Save the result
    save_image(fake_image, output_filename)

    # Optional: Display images for comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(tf.cast(tf.squeeze(hr_image), tf.uint8))
    plt.title("Original Image (Low Resolution)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(tf.cast(fake_image, tf.uint8))
    plt.title("Enhanced Image (Super Resolution)")
    plt.axis('off')
    plt.show()

# --- Example Usage ---
# NOTE: Replace 'your_low_res_image.jpg' with the actual path to your image file.
# The image resolution will be scaled up by 4x.
# enhance_image_esrgan('your_low_res_image.jpg')