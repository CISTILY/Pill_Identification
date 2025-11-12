import os
from PIL import Image
import pillow_heif
import argparse

def convert_HEIC_to_JPG(input_folder: str, output_folder: str):
    """
    Recursively finds all .HEIC files in an input directory, converts
    them to .JPG, and saves them in a matching directory structure in the
    output folder.

    It preserves the original folder hierarchy. For example, a file at:
    `input_folder/subfolder/image.heic`
    will be converted and saved to:
    `output_folder/subfolder/image.jpg`

    If the output subfolders do not exist, they will be created.
    The function will skip conversion if a .jpg file with the
    target name already exists in the destination.

    Args:
        input_folder (str): The root directory to search for .HEIC files.
        output_folder (str): The root directory where the converted .JPG
                             files (and their folder structure) will be saved.
    """
    
    print(f"Starting conversion from '{input_folder}' to '{output_folder}'...")

    # os.walk recursively explores the directory tree
    for dirpath, dirnames, filenames in os.walk(input_folder):
        
        # --- 1. Create Matching Output Directory ---
        
        # Get the relative path from the input root
        # e.g., "input_folder/vacation/photos" -> "vacation/photos"
        relative_path = os.path.relpath(dirpath, input_folder)
        
        # Define the corresponding save directory in the output folder
        # e.g., "output_folder/vacation/photos"
        # Note: If relative_path is '.', save_dir becomes 'output_folder'
        if relative_path == ".":
            save_dir = output_folder
        else:
            save_dir = os.path.join(output_folder, relative_path)

        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # --- 2. Process Files in the Current Directory ---
        for name in filenames:
            # Check if the file is a .heic file (case-insensitive)
            if name.lower().endswith(".heic"):
                
                # --- 3. Define File Paths ---
                
                # Full path to the original .heic file
                heic_path = os.path.join(dirpath, name)
                
                # Create the new .jpg filename
                # e.g., "image.HEIC" -> "image.jpg"
                jpg_filename = os.path.splitext(name)[0] + ".jpg"
                
                # Full path for the converted .jpg file
                jpg_path = os.path.join(save_dir, jpg_filename)
                
                # --- 4. Convert If Not Already Converted ---
                
                # Check if the .jpg file already exists to avoid re-work
                if not os.path.exists(jpg_path):
                    print(f"📂 Converting {heic_path} -> {jpg_path}")
                    try:
                        # Read the HEIC file using pillow_heif
                        heif_file = pillow_heif.read_heif(heic_path)

                        # Convert the HEIF data to a PIL Image object
                        image = Image.frombytes(
                            heif_file.mode,
                            heif_file.size,
                            heif_file.data,
                            "raw",
                        )

                        # Save the PIL Image as a JPG
                        image.save(jpg_path, "JPG")
                        
                    except Exception as e:
                        print(f"❌ ERROR converting {name}: {e}")
                        
                else:
                    # Skip if the file is already there
                    print(f"✅ Skipping {jpg_filename} (already exists)")
                    continue

    print("✅ Conversion complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Directory of raw HEIC images', required=True)
    parser.add_argument('--output', type=str, help='Directory of result JPG images', required=True)

    configs = parser.parse_args()
    print(configs)

    convert_HEIC_to_JPG(configs.input, configs.output)

if __name__ == "__main__":
    main()