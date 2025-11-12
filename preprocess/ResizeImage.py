import os
from PIL import Image
import argparse

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

def resize_images(input_folder: str, output_folder: str, size: int):
    """
    Recursively finds all images in `input_folder`, resizes them to a
    square `(size, size)`, and saves them to `output_folder`,
    preserving the original subdirectory structure.

    Example:
    An image at `input_folder/class_A/image_01.jpg`
    will be saved as `output_folder/class_A/image_01.jpg`.

    Args:
        input_folder (str): The root directory to search for images.
        output_folder (str): The root directory to save resized images.
        size (int): The target size (width and height) for the resized image.

    Global-Vars-Required:
        image_extensions (tuple): A tuple of file extensions (e.g., ".png", ".jpg")
                                  to process. Must be defined globally.
    """
    
    print(f"Starting resize... \nInput: {input_folder}\nOutput: {output_folder}")
    
    # os.walk explores the directory tree from the top down
    for root, _, files in os.walk(input_folder):
        
        # --- 1. Determine the relative path ---
        # This is the path *within* the input_folder
        # e.g., if root = "input/class_A", relative_path = "class_A"
        relative_path = os.path.relpath(root, input_folder)

        # --- 2. Create the corresponding output directory ---
        # If relative_path is "class_A", this creates "output/class_A"
        # If relative_path is ".", this just becomes the output_folder
        if relative_path == ".":
            current_output_dir = output_folder
        else:
            current_output_dir = os.path.join(output_folder, relative_path)
        
        # Create the directory (and any parents) if it doesn't exist
        os.makedirs(current_output_dir, exist_ok=True)
            
        # --- 3. Process all images in the current directory ---
        for filename in files:
            # Check if the file has a valid image extension
            if filename.lower().endswith(image_extensions):
                
                # Define the full path to the original image
                img_path = os.path.join(root, filename)
                
                # Define the full path where the resized image will be saved
                output_path = os.path.join(current_output_dir, filename)

                try:
                    # Open the original image
                    with Image.open(img_path) as img:
                        
                        # Resize the image using high-quality downsampling
                        # Image.LANCZOS (or Image.Resampling.LANCZOS for newer Pillow)
                        # is recommended for preserving detail.
                        # 
                        img_resized = img.resize((size, size), Image.LANCZOS)
                        
                        # Save the resized image to the output path
                        img_resized.save(output_path)
                        
                        # print(f"Resized: {img_path} -> {output_path}") # (Optional: too verbose)

                except Exception as e:
                    # Catch potential errors (e.g., corrupted files)
                    print(f"❌ Error processing {img_path}: {e}")

    print("--- \n✅ Resize complete! ---")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Directory of input images', required=True)
    parser.add_argument('-s', '--size', type=int, help='Size of result images', required=True)
    
    configs=parser.parse_args()
    print(configs)

    resize_images(configs.input, configs.size)

if __name__ == "__main__":
    main()