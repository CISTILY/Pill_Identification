import os
from PIL import Image
import argparse

def resize_images(input_folder, size):
    """
    Recursively resize all images in input_folder to given size
    and save them into output_folder (all in one folder, no subfolders).
    Keeps original filenames, avoids overwriting by adding suffix only if needed.
    """
    """
    Recursively resize all images in input_folder to given size
    and save them into output_folder, keeping the same folder structure.
    """
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                img_path = os.path.join(root, filename)

                try:
                    with Image.open(img_path) as img:
                        img_resized = img.resize((size, size), Image.LANCZOS)
                        img_resized.save(img_path)
                        print(f"Resized: {img_path} -> {img_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Directory of input images', required=True)
    parser.add_argument('-s', '--size', type=int, help='Size of result images', required=True)
    
    configs=parser.parse_args()
    print(configs)

    resize_images(configs.input, configs.size)

__main__()