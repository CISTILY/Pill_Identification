import os
import shutil
import argparse

def build_dataset(input_folders, output_folder, concatenate=False):
    os.makedirs(output_folder, exist_ok=True)

    max_id = 0
    if (concatenate == True):
        for f in os.listdir(output_folder):
            if max_id <= int(f):
                max_id = int(f)
            else:
                continue
        current_id = max_id
    else:
        current_id = 0

    img_path_list = []
    for root, _, files in os.walk(input_folders):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                img_path_list.append(os.path.join(root, filename))

    # Get all jpg files and sort them to keep order consistent
    files = [f for f in img_path_list if f.lower().endswith('.jpg')]
    files.sort()

    id_image_count = 0  # count images within this id (to know when to reset)
    group_count = 0     # used to switch MT/MS every 3 images

    # separate counters for MT and MS
    mt_counter = 0
    ms_counter = 0

    for idx, filename in enumerate(files):
        # Determine MT/MS every 3 images
        group_type = "MT" if (group_count // 3) % 2 == 0 else "MS"

        if group_type == "MT":
            mt_counter += 1
            number_in_group = mt_counter
        else:
            ms_counter += 1
            number_in_group = ms_counter

        new_filename = f"{current_id}_{group_type}_{number_in_group:02d}.jpg"

        # Create folder structure <id>/<group_type>/Mau
        id_folder = os.path.join(output_folder, str(current_id))
        group_folder = os.path.join(id_folder, group_type, "Mau")
        os.makedirs(group_folder, exist_ok=True)

        # Copy (or move) file
        src = filename
        dst = os.path.join(group_folder, new_filename)
        shutil.copy2(src, dst)  # <-- use shutil.move if you want to move instead

        # Update counters
        id_image_count += 1
        group_count += 1

        # Reset after 18 images (new id -> restart all counters)
        if id_image_count == 18:
            current_id += 1
            id_image_count = 0
            group_count = 0
            mt_counter = 0
            ms_counter = 0

    print("✅ Done! MT/MS numbers are cumulative per group type.")

def __main__():
    # Parameters parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='One or list of path to the converted image', required=True)
    parser.add_argument('--output', help='Output directory for result dataset', required= True)
    parser.add_argument('-c', '--concat', type=bool, help='One or list of path to the directory that need to add to the dataset', required=False)
    parser.add_argument('-n', '--number_of_samples', type=int, help='Number of samples per pills', required=False)

    configs = parser.parse_args()
    print(configs)

    build_dataset(configs.input, configs.output, configs.concat)

__main__()