import os
import shutil
import argparse

FLATTEN_DIR = "D:\Pill_Identification\Data\LabeledData\Images"

def build_dataset(input_folders: str, output_folder: str, flatten: bool, concatenate: bool = False):
    """
    Organizes image files from a source directory into a new, structured
    directory, renaming them based on a specific batching and grouping logic.

    The logic processes 18 images at a time, assigning them a unique ID.
    Within each 18-image batch, it alternates:
    - 9 images are labeled "MT"
    - 9 images are labeled "MS"

    The files are renamed as: <id>_<group_type>_<number>.jpg
    The <number> is a counter that increments *separately* for MT and MS
    (e.g., 0_MT_01, 0_MT_02, ..., 0_MS_01, 0_MS_02, ...).

    The final folder structure is:
    <output_folder>/<id>/<group_type>/Mau/<filename.jpg>

    Args:
        input_folders (str): The *single* root directory to recursively
                             search for images.
        output_folder (str): The root directory where the new
                             structured folders will be created.
        flatten (bool): If True, a *copy* of every renamed image will
                        also be saved in a single flat directory
                        (defined by the global variable `FLATTEN_DIR`).
        concatenate (bool, optional): If True, finds the highest existing
                                     numeric ID (folder name) in the
                                     `output_folder` and starts new IDs
                                     from there. If False, starts from ID 0.
                                     Defaults to False.
    """
    # Ensure the main output directory exists
    os.makedirs(output_folder, exist_ok=True)

    max_id = 0
    # --- Determine the starting ID ---
    if concatenate:
        # If concatenating, find the highest existing ID in the output folder
        print(f"Concatenate mode: Checking for existing IDs in {output_folder}...")
        for f in os.listdir(output_folder):
            # Check if the folder name is a digit (to avoid .DS_Store, etc.)
            if f.isdigit():
                if max_id <= int(f):
                    max_id = int(f) + 1 # Start from the next available ID
        
        current_id = max_id
        if current_id > 0:
            print(f"Found existing IDs. Starting new IDs from {current_id}.")
        else:
            print("No existing numeric folders found. Starting from ID 0.")
    else:
        # Otherwise, start from ID 0
        current_id = 0

    # --- Find all images recursively ---
    img_path_list = []
    print(f"Scanning for images in {input_folders}...")
    for root, _, files in os.walk(input_folders):
        for filename in files:
            # Check for common image extensions
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                img_path_list.append(os.path.join(root, filename))

    # --- Filter for .jpg files and sort ---
    # The logic specifically processes .jpg files based on this filter
    files = [f for f in img_path_list if f.lower().endswith('.jpg')]
    files.sort()  # Sort to ensure a consistent, predictable order
    print(f"Found {len(files)} .jpg files to process.")

    # --- Initialize counters for batching logic ---
    id_image_count = 0  # Tracks images per ID (resets every 18)
    group_count = 0     # Tracks image index within the current ID batch (0-17)
    mt_counter = 0      # Separate counter for numbering "MT" files
    ms_counter = 0      # Separate counter for numbering "MS" files

    # --- Process each file ---
    for idx, filename in enumerate(files):
        
        # --- Determine Group Type (MT/MS) ---
        # (group_count // 3) creates groups of 3:
        #   (0,1,2)//3 = 0
        #   (3,4,5)//3 = 1
        #   (6,7,8)//3 = 2
        # (% 2) alternates these groups:
        #   0 % 2 = 0 (MT)
        #   1 % 2 = 1 (MS)
        #   2 % 2 = 0 (MT)
        # Result: 3 images MT, 3 images MS, 3 images MT, etc.
        group_type = "MT" if (group_count // 3) % 2 == 0 else "MS"

        # Increment the correct counter and get the number for the filename
        if group_type == "MT":
            mt_counter += 1
            number_in_group = mt_counter
        else:
            ms_counter += 1
            number_in_group = ms_counter

        # Format the new filename, padding the number to 2 digits (e.g., 01, 02)
        new_filename = f"{current_id}_{group_type}_{number_in_group:02d}.jpg"

        # --- Create the nested folder structure ---
        # e.g., output_folder/0/MT/Mau/
        id_folder = os.path.join(output_folder, str(current_id))
        group_folder = os.path.join(id_folder, group_type, "Mau")
        os.makedirs(group_folder, exist_ok=True)

        # --- Copy the file ---
        src = filename
        dst = os.path.join(group_folder, new_filename)
        shutil.copy2(src, dst)  # copy2 preserves metadata (like creation time)

        # If flatten is enabled, copy to the flat directory as well
        if flatten:
            try:
                dst_flatten = os.path.join(FLATTEN_DIR, new_filename)
                shutil.copy2(src, dst_flatten)
            except NameError:
                print(f"Error: 'FLATTEN_DIR' is not defined. Cannot flatten {new_filename}.")
            except Exception as e:
                print(f"Error flattening {new_filename}: {e}")


        # --- Update counters for the 18-image batch ---
        id_image_count += 1
        group_count += 1

        # --- Reset after 18 images ---
        # This signals the end of one "ID" group
        if id_image_count == 18:
            print(f"Completed ID {current_id} (18 images processed).")
            # Move to the next ID
            current_id += 1
            # Reset all counters for the new batch
            id_image_count = 0
            group_count = 0
            mt_counter = 0
            ms_counter = 0

    print(f"✅ Done! Processed {len(files)} files.")

def main():
    # Parameters parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='One or list of path to the converted image', required=True)
    parser.add_argument('--output', help='Output directory for result dataset', required= True)
    parser.add_argument('-f', '--flat', type=bool, help='Create flattend dataset (used for model training)', required=False)
    parser.add_argument('-c', '--concat', type=bool, help='One or list of path to the directory that need to add to the dataset', required=False)
    parser.add_argument('-n', '--number_of_samples', type=int, help='Number of samples per pills', required=False)

    configs = parser.parse_args()
    print(configs)

    build_dataset(configs.input, configs.output, configs.flat, configs.number_of_samples)

if __name__ == "__main__":
    main()