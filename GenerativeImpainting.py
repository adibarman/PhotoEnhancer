#GG
import torch
import requests
from io import BytesIO
from PIL import Image
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# --- 1. Model Initialization ---
# The 'runwayml/stable-diffusion-inpainting' is a model specifically fine-tuned for inpainting.
# We use AutoPipelineForInpainting to load the correct pipeline class.
print("Loading Stable Diffusion Inpainting model...")

# Use 'cuda' if GPU is available, otherwise use 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Load model with half-precision (FP16) on GPU for speed and memory efficiency
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    # Enable CPU offload to save VRAM on smaller GPUs
    pipeline.enable_model_cpu_offload() 

except Exception as e:
    print(f"GPU/FP16 load failed ({e}). Falling back to CPU/FP32...")
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting"
    ).to("cpu")


def run_inpainting(original_image_path, mask_image_path, prompt, output_filename="inpainted_result.png"):
    """
    Performs text-guided inpainting on a masked area of an image.

    Args:
        original_image_path (str): Path to the image to be edited.
        mask_image_path (str): Path to the mask image (white pixels = fill, black pixels = keep).
        prompt (str): Text prompt describing what should fill the masked area.
    """
    
    # 2. Load Images
    # Note: Images are often resized to 512x512 internally for the model
    try:
        init_image = load_image(original_image_path).convert("RGB").resize((512, 512))
        mask_image = load_image(mask_image_path).convert("RGB").resize((512, 512))
    except Exception as e:
        print(f"Error loading images: {e}. Please check paths.")
        return

    print(f"Starting inpainting with prompt: '{prompt}'")

    # 3. Run the Pipeline
    generated_image = pipeline(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        # Hyperparameters for control
        guidance_scale=7.5,      # How strongly the prompt guides the generation
        num_inference_steps=50   # Quality/time tradeoff
    ).images[0]
    
    # 4. Save Result
    generated_image.save(output_filename)
    print(f"✅ Inpainting complete. Saved to {output_filename}")

# --- Example Usage (Requires you to create two local files: original.jpg and mask.png) ---

# 1. Create 'original.jpg' (your photo).
# 2. Create 'mask.png' which is a black and white image:
#    - Areas you want the AI to generate new content *in* should be **WHITE**.
#    - Areas you want to *keep* unchanged should be **BLACK**.

# Example (Uncomment to run after setting up your files):
# run_inpainting(
#     original_image_path="original.jpg",
#     mask_image_path="mask.png",
#     prompt="A vibrant orange sunset over a tropical beach, highly detailed",
#     output_filename="generative_enhance.png"
# )