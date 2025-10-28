


import os
# ... (other imports from the previous code) ...

def enhance_photo(client, file_path):
    """Core function to enhance a single photo."""
    # ... (content of the enhance_photo_with_gemini function, adapted) ...
    try:
        # Load the image
        img = Image.open(file_path)

        # Call the model with the image and prompt
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, ENHANCEMENT_PROMPT]
        )

        if response.candidates and response.candidates[0].image:
            enhanced_image_bytes = response.candidates[0].image.image_bytes
            enhanced_img = Image.open(BytesIO(enhanced_image_bytes))
            
            # Construct the output file path (e.g., 'enhanced_photo_1.png')
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"enhanced_{file_name.replace('.', '_')}.png")
            
            enhanced_img.save(output_path)
            print(f"✅ Enhanced {file_name} -> {output_path}")
        else:
            print(f"❌ Failed to enhance {file_name}. No image returned.")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")


def batch_enhance_photos(input_dir, output_dir):
    """Iterates through a directory and enhances all images."""
    os.makedirs(output_dir, exist_ok=True)
    client = genai.Client()
    
    print(f"Starting batch enhancement in {input_dir}...")
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_dir, filename)
            enhance_photo(client, file_path)
            
    print("\nBatch enhancement complete.")

# --- Define Directories ---
input_dir = "photos_to_enhance"
output_dir = "enhanced_photos"

# Uncomment the line below to run the batch process
# if __name__ == "__main__":
#     batch_enhance_photos(input_dir, output_dir)