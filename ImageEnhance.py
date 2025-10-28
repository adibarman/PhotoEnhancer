#CA

# /*

# # Auto-enhance all images in a folder
# python image_enhancer.py /path/to/images

# # Specify custom output folder
# python image_enhancer.py /path/to/images -o /path/to/output

# # Process subfolders recursively
# python image_enhancer.py /path/to/images -r

# # Use manual settings
# python image_enhancer.py /path/to/images --manual --brightness 0.2 --contrast 0.3 --saturation 0.15

# # Custom prefix for output files
# python image_enhancer.py /path/to/images -p "AI_enhanced_"

# */

# /*

# from image_enhancer import ImageEnhancer

# enhancer = ImageEnhancer()

# # Process entire folder with auto-enhance
# enhancer.batch_enhance('/path/to/images')

# # Process single image
# enhancer.enhance_image('photo.jpg', 'enhanced_photo.jpg')

# # Manual settings
# enhancer.enhance_image('photo.jpg', 'enhanced.jpg', auto=False,
#                        brightness=0.2, contrast=0.3, saturation=0.15)

# */


import os
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
import argparse
from datetime import datetime


class ImageEnhancer:
    """AI-powered image enhancement similar to Google Photos"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def analyze_image(self, img_array):
        """Analyze image characteristics to determine optimal enhancements"""
        # Normalize to 0-255 range
        if img_array.max() <= 1.0:
            img_array = img_array * 255
        
        # Calculate statistics
        brightness = np.mean(img_array)
        
        # Calculate saturation (based on RGB color space)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = np.mean(np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0))
        
        # Calculate histogram for highlights/shadows analysis
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        total_pixels = gray.size
        
        dark_pixels = np.sum(hist[:64]) / total_pixels
        bright_pixels = np.sum(hist[192:]) / total_pixels
        
        # Determine adjustments
        adjustments = {
            'brightness': (128 - brightness) / 255 * 0.3,
            'contrast': 0.2 if (brightness < 100 or brightness > 155) else 0.1,
            'saturation': 0.2 if saturation < 0.3 else 0,
            'highlights': -0.2 if bright_pixels > 0.2 else 0,
            'shadows': 0.3 if dark_pixels > 0.2 else 0,
            'sharpness': 0.3,
            'warmth': 0,
            'vignette': 0.1
        }
        
        return adjustments
    
    def apply_brightness(self, img_array, amount):
        """Adjust brightness"""
        if amount == 0:
            return img_array
        adjust = amount * 50
        return np.clip(img_array + adjust, 0, 255)
    
    def apply_contrast(self, img_array, amount):
        """Adjust contrast"""
        if amount == 0:
            return img_array
        factor = 1 + amount
        return np.clip(((img_array - 128) * factor) + 128, 0, 255)
    
    def apply_saturation(self, img_array, amount):
        """Adjust saturation"""
        if amount == 0:
            return img_array
        
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        factor = 1 + amount
        result = np.zeros_like(img_array)
        result[:, :, 0] = np.clip(gray + (r - gray) * factor, 0, 255)
        result[:, :, 1] = np.clip(gray + (g - gray) * factor, 0, 255)
        result[:, :, 2] = np.clip(gray + (b - gray) * factor, 0, 255)
        
        return result
    
    def apply_highlights_shadows(self, img_array, highlights, shadows):
        """Adjust highlights and shadows separately"""
        if highlights == 0 and shadows == 0:
            return img_array
        
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        brightness = (r + g + b) / 3
        
        result = img_array.copy()
        
        # Adjust highlights
        if highlights != 0:
            highlight_mask = (brightness > 170).astype(float)
            factor = 1 - (highlights * 0.5)
            for i in range(3):
                result[:, :, i] = np.clip(
                    result[:, :, i] * (1 - highlight_mask + highlight_mask * factor),
                    0, 255
                )
        
        # Adjust shadows
        if shadows != 0:
            shadow_mask = (brightness < 85).astype(float)
            factor = 1 + (shadows * 0.5)
            for i in range(3):
                result[:, :, i] = np.clip(
                    result[:, :, i] * (1 - shadow_mask + shadow_mask * factor),
                    0, 255
                )
        
        return result
    
    def apply_warmth(self, img_array, amount):
        """Adjust color temperature (warmth)"""
        if amount == 0:
            return img_array
        
        result = img_array.copy()
        result[:, :, 0] = np.clip(result[:, :, 0] + amount * 20, 0, 255)  # Red
        result[:, :, 2] = np.clip(result[:, :, 2] - amount * 20, 0, 255)  # Blue
        
        return result
    
    def apply_sharpness(self, img, amount):
        """Apply sharpening filter"""
        if amount <= 0:
            return img
        
        # Convert amount to a reasonable range for PIL
        # PIL's SHARPEN filter is binary, so we'll use UnsharpMask for more control
        radius = 2
        percent = int(amount * 150)  # Scale to percentage
        threshold = 3
        
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    
    def apply_vignette(self, img_array, amount):
        """Apply vignette effect"""
        if amount <= 0:
            return img_array
        
        h, w = img_array.shape[:2]
        center_x, center_y = w / 2, h / 2
        
        # Create coordinate grids
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        # Calculate distance from center
        dx = xx - center_x
        dy = yy - center_y
        distance = np.sqrt(dx**2 + dy**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Create vignette mask
        vignette_mask = 1 - (distance / max_dist) * amount
        vignette_mask = np.clip(vignette_mask, 0, 1)
        
        # Apply to all channels
        result = img_array.copy()
        for i in range(3):
            result[:, :, i] = np.clip(result[:, :, i] * vignette_mask, 0, 255)
        
        return result
    
    def enhance_image(self, img_path, output_path=None, auto=True, **manual_settings):
        """
        Enhance a single image
        
        Args:
            img_path: Path to input image
            output_path: Path to save enhanced image (optional)
            auto: Use automatic enhancement (True) or manual settings (False)
            **manual_settings: Manual adjustment values if auto=False
        """
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            
            # Determine settings
            if auto:
                settings = self.analyze_image(img_array)
            else:
                settings = {
                    'brightness': manual_settings.get('brightness', 0),
                    'contrast': manual_settings.get('contrast', 0),
                    'saturation': manual_settings.get('saturation', 0),
                    'highlights': manual_settings.get('highlights', 0),
                    'shadows': manual_settings.get('shadows', 0),
                    'warmth': manual_settings.get('warmth', 0),
                    'sharpness': manual_settings.get('sharpness', 0),
                    'vignette': manual_settings.get('vignette', 0)
                }
            
            # Apply enhancements
            result = img_array.copy()
            result = self.apply_brightness(result, settings['brightness'])
            result = self.apply_contrast(result, settings['contrast'])
            result = self.apply_saturation(result, settings['saturation'])
            result = self.apply_highlights_shadows(result, settings['highlights'], settings['shadows'])
            result = self.apply_warmth(result, settings['warmth'])
            result = self.apply_vignette(result, settings['vignette'])
            
            # Convert back to PIL Image for sharpening
            result_img = Image.fromarray(result.astype(np.uint8))
            result_img = self.apply_sharpness(result_img, settings['sharpness'])
            
            # Save or return
            if output_path:
                result_img.save(output_path, quality=95)
                return output_path
            else:
                return result_img
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None
    
    def batch_enhance(self, input_folder, output_folder=None, recursive=False, 
                     prefix='enhanced_', auto=True, **manual_settings):
        """
        Enhance all images in a folder
        
        Args:
            input_folder: Path to folder containing images
            output_folder: Path to save enhanced images (defaults to input_folder/enhanced/)
            recursive: Process subfolders recursively
            prefix: Prefix for enhanced image filenames
            auto: Use automatic enhancement
            **manual_settings: Manual adjustment values if auto=False
        """
        input_path = Path(input_folder)
        
        if not input_path.exists():
            print(f"Error: Input folder '{input_folder}' does not exist")
            return
        
        # Set up output folder
        if output_folder is None:
            output_path = input_path / 'enhanced'
        else:
            output_path = Path(output_folder)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        if recursive:
            image_files = []
            for ext in self.supported_formats:
                image_files.extend(input_path.rglob(f'*{ext}'))
                image_files.extend(input_path.rglob(f'*{ext.upper()}'))
        else:
            image_files = []
            for ext in self.supported_formats:
                image_files.extend(input_path.glob(f'*{ext}'))
                image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in '{input_folder}'")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Output folder: {output_path}")
        print("-" * 50)
        
        # Process each image
        processed = 0
        failed = 0
        start_time = datetime.now()
        
        for idx, img_file in enumerate(image_files, 1):
            # Create output path preserving subfolder structure if recursive
            if recursive:
                rel_path = img_file.relative_to(input_path)
                out_file = output_path / rel_path.parent / f"{prefix}{rel_path.name}"
                out_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_file = output_path / f"{prefix}{img_file.name}"
            
            print(f"[{idx}/{len(image_files)}] Processing: {img_file.name}...", end=' ')
            
            result = self.enhance_image(img_file, out_file, auto=auto, **manual_settings)
            
            if result:
                print("✓ Success")
                processed += 1
            else:
                print("✗ Failed")
                failed += 1
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        print("-" * 50)
        print(f"Processing complete!")
        print(f"Total: {len(image_files)} images")
        print(f"Successful: {processed}")
        print(f"Failed: {failed}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Average: {elapsed/len(image_files):.2f} seconds per image")


def main():
    parser = argparse.ArgumentParser(description='AI Image Enhancer - Batch process images')
    parser.add_argument('input_folder', help='Folder containing images to enhance')
    parser.add_argument('-o', '--output', help='Output folder (default: input_folder/enhanced/)')
    parser.add_argument('-r', '--recursive', action='store_true', help='Process subfolders recursively')
    parser.add_argument('-p', '--prefix', default='enhanced_', help='Prefix for output filenames')
    parser.add_argument('--manual', action='store_true', help='Use manual settings instead of auto-enhance')
    
    # Manual adjustment arguments
    parser.add_argument('--brightness', type=float, default=0, help='Brightness adjustment (-1 to 1)')
    parser.add_argument('--contrast', type=float, default=0, help='Contrast adjustment (-1 to 1)')
    parser.add_argument('--saturation', type=float, default=0, help='Saturation adjustment (-1 to 1)')
    parser.add_argument('--highlights', type=float, default=0, help='Highlights adjustment (-1 to 1)')
    parser.add_argument('--shadows', type=float, default=0, help='Shadows adjustment (-1 to 1)')
    parser.add_argument('--warmth', type=float, default=0, help='Color temperature adjustment (-1 to 1)')
    parser.add_argument('--sharpness', type=float, default=0, help='Sharpness adjustment (0 to 1)')
    parser.add_argument('--vignette', type=float, default=0, help='Vignette amount (0 to 1)')
    
    args = parser.parse_args()
    
    enhancer = ImageEnhancer()
    
    manual_settings = {
        'brightness': args.brightness,
        'contrast': args.contrast,
        'saturation': args.saturation,
        'highlights': args.highlights,
        'shadows': args.shadows,
        'warmth': args.warmth,
        'sharpness': args.sharpness,
        'vignette': args.vignette
    }
    
    enhancer.batch_enhance(
        args.input_folder,
        output_folder=args.output,
        recursive=args.recursive,
        prefix=args.prefix,
        auto=not args.manual,
        **manual_settings
    )


if __name__ == '__main__':
    main()

