from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

# --- Configuration ---
WATERMARK_TEXT = "© Aditya B Sept,2025"

WATERMARK_COLOR_BRIGHT_YELLOW = (255, 255, 0, 128)
WATERMARK_COLOR_WHITE = (255, 255, 255, 128)
WATERMARK_COLOR_ORANGE_YELLOW = (255, 176, 0, 128)
WATERMARK_COLOR_LIGHT_GRAY = (200, 200, 200, 128)
FONT_PATH = "arial.ttf"
FONT_SIZE = 40
#Make the font bold by using a bold font file if available
BOLD_FONT_PATH = "arialbd.ttf"

def add_watermark(input_image_path, output_image_path, text, color, position, font_path=BOLD_FONT_PATH, font_size=FONT_SIZE):
    # Open image
    img = Image.open(input_image_path).convert("RGBA")

    # Make a transparent layer
    txt = Image.new("RGBA", img.size, (255,255,255,0))

    # Draw watermark
    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype(font_path, font_size)

    draw.text(position, text, fill=color, font=font)

    # Combine and save
    watermarked = Image.alpha_composite(img, txt)
    watermarked.convert("RGB").save(output_image_path, "JPEG")



if __name__ == "__main__":

    # Input image file name
    #input_image = "2025PreFall.jpg"

    # Input Folder path
    input_folder = Path("C:/Adi/Photo/FB")

    # Output Folder path
    output_folder = input_folder / "Watermarked"
    # Create the output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    #Add watermark to all the images in the input folder
    for input_image in os.listdir(input_folder):
        if not input_image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue  # Skip non-image files
        print(f"Processing image: {input_image}")
        input_image_path = input_folder / input_image
        print(f"Input image path: {input_image_path}")
        # Output image file name
        output_image = "WM_" + input_image
        output_image_path = output_folder / output_image

        # Open image to get dimensions
        img = Image.open(input_folder / input_image)
        width, height = img.size
        print(f"Image dimensions: {width}x{height}")
        
        #Text position bottom left corner.
        text_position = (10, height - 70)


    add_watermark(
        input_image_path=input_folder / input_image,
        output_image_path=output_image_path,
        text=WATERMARK_TEXT,
        color=WATERMARK_COLOR_WHITE, # <-- Easily change the color here
        position=text_position
    )

    print(f"Watermarked image saved as {output_image_path}")
