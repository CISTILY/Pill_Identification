import os
from PIL import Image
import pillow_heif
import argparse

def convert_HEIC_to_JPG(input_folder, output_folder): 
    # Iterate over all .HEIC files
    print("Listing images")
    for dirpath, dirnames, filenames in os.walk(input_folder):
        relative_path = os.path.relpath(dirpath, input_folder)
        print(relative_path)
        relative_path = relative_path.replace("HEIC", "JPG")
        save_dir = os.path.join(output_folder, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        for name in filenames:
            if name.lower().endswith(".heic"):
                heic_path = os.path.join(dirpath, name)
                jpg_filename = os.path.splitext(name)[0] + ".jpg"
                jpg_path = os.path.join(save_dir, jpg_filename)
                print(heic_path)
                print(jpg_filename)
                print(jpg_path)
                if os.path.exists(jpg_path) is False:
                    print(f"📂 Converting {name} -> {jpg_filename}")

                    # Read HEIC and convert
                    heif_file = pillow_heif.read_heif(heic_path)
                    image = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw",
                    )

                    image.save(jpg_path)
                else:
                    print(f"file is already exist. Skipping")
                    continue

    print("✅ Conversion complete!")

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Directory of raw HEIC images', required=True)
    parser.add_argument('--output', type=str, help='Directory of result JPG images', required=True)

    configs = parser.parse_args()
    print(configs)

    convert_HEIC_to_JPG(configs.input, configs.output)

__main__()