#pip install google-genai pillow

#Bash
#export GEMINI_API_KEY="YOUR_API_KEY"

import os
from google import genai
from PIL import Image
from io import BytesIO

# --- Configuration ---
# The prompt is the key to generative enhancement.
# Be descriptive about the desired improvements!
ENHANCEMENT_PROMPT = (
    "Please enhance this photo to make it look professional and vibrant. "
    "The final image should have perfect exposure, natural colors, "
    "reduced noise, increased sharpness, and cinematic lighting."
)
INPUT_FILE = "input_photo.jpg"
OUTPUT_FILE = "enhanced_photo.png"
MODEL_NAME = "gemini-2.5-flash"

def enhance_photo_with_gemini():
    """
    Uses the Gemini API to enhance a photo based on a text prompt.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found. Please create one.")
        return

    try:
        # 1. Initialize the client
        client = genai.Client()

        # 2. Load the image
        img = Image.open(INPUT_FILE)
        print(f"Loaded image: {INPUT_FILE} ({img.size[0]}x{img.size[1]})")

        # 3. Call the model with the image and prompt
        print(f"Sending enhancement request to {MODEL_NAME} with prompt: '{ENHANCEMENT_PROMPT[:50]}...'")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[img, ENHANCEMENT_PROMPT]
        )

        # 4. Extract and save the enhanced image
        if response.candidates and response.candidates[0].image:
            # The enhanced image is in the response.candidates[0].image object
            enhanced_image_bytes = response.candidates[0].image.image_bytes
            
            # Convert bytes back to a PIL Image
            enhanced_img = Image.open(BytesIO(enhanced_image_bytes))
            
            # Save the result
            enhanced_img.save(OUTPUT_FILE)
            print(f"\n✅ Success! Enhanced image saved to {OUTPUT_FILE}")
        else:
            print("❌ Enhancement failed. No image candidate returned.")
            # Optionally print the text response for debugging
            # print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    enhance_photo_with_gemini()


    