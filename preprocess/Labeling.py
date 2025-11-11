import cv2
import numpy as np
import os
import shutil
import argparse
import csv

image_extensions = ('.png', '.jpg')

def create_label(input_folder, output_folder, type, remove=False):
    csv_output_path = None
    txt_labels_dir = None
    txt_bbox_dir = None

    if type == ".txt":
        os.makedirs(os.path.join(output_folder, "TXT", "Labels"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "TXT", "BBox"), exist_ok=True)

    elif type == ".csv":
        os.makedirs(os.path.join(output_folder, "CSV", "Labels"), exist_ok=True)
        csv_output_path = os.path.join(output_folder, "CSV", "Labels", "Labels.csv")

        try:
            with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)

                csv_writer.writerow(['class_id', 'width', 'height', 'x', 'y', 'w', 'h'])
        except IOError as e:
            print("Error when creating CSV: {e}")
            return
    
    else:
        print("Format not supported")
        return

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                input_path = os.path.join(root, file)

                # Filename without extension
                filename_wo_ext = os.path.splitext(file)[0]

                # Load mask
                mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"❌ Failed to load image: {input_path}")
                    continue

                # Threshold to binary
                _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                height, width = binary.shape
                output_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # Extract class ID from filename prefix
                class_id = filename_wo_ext.split('_')[0]

                if type == ".txt":
                    label_lines = []
                    
                    txt_labels_dir = os.path.join(output_folder, "TXT", "Labels", filename_wo_ext + ".txt")
                    txt_bbox_dir = os.path.join(output_folder, "TXT", "BBox", filename_wo_ext + "_bbox.png")


                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)

                        # Convert to YOLO format
                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        w_norm = w / width
                        h_norm = h / height

                        yolo_format = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                        label_lines.append(yolo_format)

                        # Draw bounding box
                        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Save YOLO label file
                    with open(txt_labels_dir, "x") as f:
                        for line in label_lines:
                            f.write(line)  # ensure newline per label

                    # Save image with bounding boxes
                    cv2.imwrite(txt_bbox_dir, output_img)
                    print(f"✅ Processed: {file}")
                
                elif type == ".csv":
                    with open(csv_output_path, mode='a', newline='', encoding='utf-8') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)

                            # Convert to float
                            x_f = float(x)
                            y_f = float(y)
                            w_f = float(w)
                            h_f = float(h)

                            # Write row
                            csv_writer.writerow([filename_wo_ext, width, height, x_f, y_f, w_f, h_f])

                        print(f"✅ Processed: {input_path}")


def __main__():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', help='Input of image mask', type=str, required=True)
    parser.add_argument('--output', help='Output directory for labels and boundary box', type=str, required=True)
    parser.add_argument('-t', '--type', help="Format type of labels (.csv, .txt, ...)", type=str, required=True)
    parser.add_argument('-r', '--remove', help="Flag to indicate remove current reuslt", type=bool, required=False)

    configs = parser.parse_args()
    print(configs)

    create_label(configs.input, configs.output, configs.type, configs.remove)

__main__()