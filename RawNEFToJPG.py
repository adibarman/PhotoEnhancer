import os
import pathlib
import rawpy
import imageio.v3 as iio
from rich.progress import track

# --- Configuration ---
RAW_INPUT_DIR = "nef_raw_photos"
PROCESSED_OUTPUT_DIR = "intermediate_jpgs"

def convert_nef_to_jpg_batch(input_folder, output_folder):
    """
    Converts all NEF files in a folder to JPG, applying camera's white balance.
    """
    os.makedirs(output_folder, exist_ok=True)
    input_path = pathlib.Path(input_folder)
    
    # Find all NEF files (case-insensitive)
    images = list(input_path.glob("*.nef")) + list(input_path.glob("*.NEF"))
    
    if not images:
        print(f"No NEF files found in {input_folder}. Please check the directory.")
        return

    print(f"Found {len(images)} NEF files. Starting conversion...")

    for img_path in track(images, description="[green]Processing RAW files...[/green]"):
        try:
            with rawpy.imread(str(img_path)) as raw:
                # 1. Post-process the RAW data
                # use_camera_wb=True applies the white balance setting recorded in the NEF file
                rgb = raw.postprocess(rawpy.Params(
                    use_camera_wb=True,
                    output_bps=8  # Output 8-bit image (standard for JPG)
                ))
            
            # 2. Save the result as a JPG
            output_file_name = img_path.name.split('.')[0] + ".jpg"
            output_file_path = pathlib.Path(output_folder) / output_file_name
            
            iio.imwrite(output_file_path, rgb, plugin='pillow', quality=95) # Save with high quality
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")

if __name__ == "__main__":
    # Create the input directory for testing
    os.makedirs(RAW_INPUT_DIR, exist_ok=True)
    print(f"Please place your .NEF files into the '{RAW_INPUT_DIR}' directory.")
    
    # Run the conversion
    # convert_nef_to_jpg_batch(RAW_INPUT_DIR, PROCESSED_OUTPUT_DIR)
    
    # NOTE: You must uncomment the line above and ensure you have NEF files 
    # in the input directory before running.
    
    print(f"\nAfter conversion, the files will be in the '{PROCESSED_OUTPUT_DIR}' directory, ready for AI enhancement (Step 2).")