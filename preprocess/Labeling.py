import cv2
import numpy as np
import os
import shutil
import argparse
import csv

image_extensions = ('.png', '.jpg')

def create_label(input_folder: str, output_folder: str, type: str, remove: bool = False):
    """
    Generates bounding box labels from binary mask images.

    This function recursively walks an input directory, finds all images
    (assumed to be binary masks), and computes contours for each mask.
    It then saves the bounding box for each contour in either YOLO .txt format
    or a single .csv file.

    The class ID for all objects in a mask is derived from its
    filename, which is assumed to be in the format: "CLASSID_...rest_of_name.png".
    For example, "0_pill.png" will have a class_id of "0".

    Args:
        input_folder (str): The root directory containing the mask images.
        output_folder (str): The root directory where labels will be saved.
        type (str): The desired output format. Must be ".txt" or ".csv".
        remove (bool, optional): This parameter is not currently used
                                 in the function. Defaults to False.

    Output Structure (for .txt):
        - <output_folder>/Labels/
            - <filename_wo_ext>.txt (YOLO format labels)
        - <output_folder>/BBox/
            - <filename_wo_ext>_bbox.png (Visual copy with boxes drawn)

    Output Structure (for .csv):
        - <output_folder>/Labels/
            - Labels.csv (A single CSV file for all images)
    """
    
    # --- 1. Initialize Paths and Variables ---
    csv_output_path = None
    txt_labels_dir = None
    txt_bbox_dir = None

    # --- 2. Setup Output Directories based on Type ---
    if type == ".txt":
        # For .txt, create a folder for label files and a folder for bbox visualizations
        os.makedirs(os.path.join(output_folder, "Labels"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "BBox"), exist_ok=True)

    elif type == ".csv":
        # For .csv, create a folder and a single .csv file with a header
        os.makedirs(os.path.join(output_folder, "Labels"), exist_ok=True)
        csv_output_path = os.path.join(output_folder, "Labels", "Labels.csv")

        try:
            # Open in 'w' (write) mode to create/overwrite the file and write the header
            with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                # Write the header row
                # NOTE: The first column 'class_id' is written, but the code
                # later populates this column with 'filename_wo_ext'.
                csv_writer.writerow(['class_id', 'width', 'height', 'x', 'y', 'w', 'h'])
        except IOError as e:
            print(f"Error when creating CSV: {e}")
            return
    
    else:
        print("Format not supported. Please use '.txt' or '.csv'.")
        return

    # --- 3. Walk Through Input Directory ---
    print("Starting label generation...")
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Check if the file has one of the specified image extensions
            if file.lower().endswith(image_extensions):
                input_path = os.path.join(root, file)

                # Get the filename without its extension (e.g., "0_pill")
                filename_wo_ext = os.path.splitext(file)[0]

                # --- 4. Process Each Mask Image ---
                
                # Load the mask as grayscale
                mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"❌ Failed to load image: {input_path}")
                    continue

                # Threshold the image to ensure it's a binary (0 or 255) mask
                _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                # Find contours (outlines of the white objects)
                # 
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Get the dimensions of the mask
                height, width = binary.shape
                
                # Create a 3-channel (color) version of the mask to draw green boxes on
                output_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # --- 5. Extract Class ID ---
                # CRITICAL: Assumes filename format is "CLASSID_...*.png"
                # e.g., "0_image.png" -> class_id = "0"
                try:
                    class_id = filename_wo_ext.split('_')[0]
                except IndexError:
                    print(f"⚠️ Warning: Could not parse class_id from {file}. Skipping.")
                    continue

                # --- 6A. Handle .txt (YOLO) Output ---
                if type == ".txt":
                    label_lines = []  # To store all "class_id x_c y_c w h" strings
                    
                    # Define output paths for this specific image
                    txt_labels_dir = os.path.join(output_folder, "Labels", filename_wo_ext + ".txt")
                    txt_bbox_dir = os.path.join(output_folder, "BBox", filename_wo_ext + "_bbox.png")

                    # Iterate over every contour (object) found in the mask
                    for cnt in contours:
                        # Get the pixel-based bounding box (x, y, width, height)
                        x, y, w, h = cv2.boundingRect(cnt)

                        # --- Convert to YOLO format ---
                        # 
                        # x_center and y_center are normalized (0.0 to 1.0)
                        # w_norm and h_norm are normalized (0.0 to 1.0)
                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        w_norm = w / width
                        h_norm = h / height

                        # Create the YOLO format string
                        yolo_format = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                        label_lines.append(yolo_format)

                        # Draw the bounding box on the visualization image
                        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Save the YOLO label .txt file
                    try:
                        # 'x' = exclusive create. Fails if file exists. Use 'w' to overwrite.
                        with open(txt_labels_dir, "x") as f:
                            # Write each label line, separated by a newline
                            f.write("\n".join(label_lines))
                            
                    except FileExistsError:
                        print(f"⚠️ Label file {txt_labels_dir} already exists. Skipping.")
                    except Exception as e:
                        print(f"❌ Error writing {txt_labels_dir}: {e}")

                    # Save the visualization image with bounding boxes
                    cv2.imwrite(txt_bbox_dir, output_img)
                    print(f"✅ Processed (txt): {file}")
                
                # --- 6B. Handle .csv Output ---
                elif type == ".csv":
                    # Open the single CSV file in 'append' mode to add new rows
                    with open(csv_output_path, mode='a', newline='', encoding='utf-8') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        
                        for cnt in contours:
                            # Get the pixel-based bounding box
                            x, y, w, h = cv2.boundingRect(cnt)

                            # Convert to float for consistency in the CSV
                            x_f = float(x)
                            y_f = float(y)
                            w_f = float(w)
                            h_f = float(h)

                            # Write the row with pixel coordinates
                            # NOTE: This writes 'filename_wo_ext' in the first column,
                            # even though the header was 'class_id'.
                            csv_writer.writerow([filename_wo_ext, width, height, x_f, y_f, w_f, h_f])

                    print(f"✅ Processed (csv): {file}")
    
    print("--- \n🎉 Label generation complete! ---")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', help='Input of image mask', type=str, required=True)
    parser.add_argument('--output', help='Output directory for labels and boundary box', type=str, required=True)
    parser.add_argument('-t', '--type', help="Format type of labels (.csv, .txt, ...)", type=str, required=True)
    parser.add_argument('-r', '--remove', help="Flag to indicate remove current reuslt", type=bool, required=False)

    configs = parser.parse_args()
    print(configs)

    create_label(configs.input, configs.output, configs.type, configs.remove)

if __name__ == "__main__":
    main()