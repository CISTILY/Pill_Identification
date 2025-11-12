import ultralytics
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import warnings
import torch
import yaml
import glob
import numpy as np
from sklearn.model_selection import KFold, train_test_split # Added train_test_split
import shutil # For file operations

# --- 2. Configuration & Constants ---

# --- MUST-HAVE ---
# 1. Define the paths to your *single* data folder
ALL_IMAGES_DIR = "D:/Pill_Identification/Data/LabeledData/images"
ALL_LABELS_DIR = "D:/Pill_Identification/Data/LabeledData/labels"

# 2. Define your class names in the correct order
# (This MUST match the class IDs in your label files)
CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
NUM_CLASSES = len(CLASS_NAMES)

# 3. Define your project directory and base model
ROOT_DIR = "D:/Pill_Identification/model/YOLOv8"
BASE_MODEL = 'yolov8s.pt'
# --- END MUST-HAVE ---

# --- Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_SET_SIZE = 0.20  # 20% of data will be held back for the final test set
K_FOLDS = 5           # Number of folds for cross-validation
CONFIG_DIR = os.path.join(ROOT_DIR, 'data_splits_config') # To store generated YAML/TXT files

# Create the config directory
os.makedirs(CONFIG_DIR, exist_ok=True)

# --- 3. Functional Blocks ---

def prepare_data_splits(images_dir, labels_dir, test_size, random_state=42):
    """
    Finds all images, verifies labels, and splits them into a train_val pool
    and a final test set.
    """
    print(f"--- Preparing Data Splits ---")
    
    # Find all images (adjust extensions if needed)
    all_image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    verified_image_paths = []
    
    # Verify that each image has a corresponding label file
    for img_path in all_image_paths:
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        if os.path.exists(label_path):
            verified_image_paths.append(img_path)
        else:
            print(f"Warning: Skipping {img_path} (missing label file {label_path})")
            
    print(f"Found {len(verified_image_paths)} images with corresponding labels.")
    
    # Split the verified images into a (train + val) pool and a test set
    train_val_pool, test_set_files = train_test_split(
        verified_image_paths,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"  > Training/Validation Pool: {len(train_val_pool)} images")
    print(f"  > Final Test Set: {len(test_set_files)} images")
    
    return train_val_pool, test_set_files


def run_kfold_training(
    train_val_pool, 
    nc, 
    names, 
    base_model, 
    epochs, 
    batch_size, 
    img_size, 
    lr0, 
    lrf, 
    device, 
    k=5
):
    """
    Runs K-fold cross-validation on the provided 'train_val_pool' of images.
    """
    print(f"\n--- Starting {k}-Fold Cross-Validation ---")
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_metrics_map50 = []
    
    kfold_project_dir = os.path.join(ROOT_DIR, 'runs', 'kfold_detect')
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(train_val_pool)):
        fold_num = fold_idx + 1
        print(f"\n--- Fold {fold_num}/{k} ---")
        
        # Get file paths for this fold
        train_files = [train_val_pool[i] for i in train_indices]
        val_files = [train_val_pool[i] for i in val_indices]
        
        # --- Write train.txt and val.txt for this fold ---
        fold_config_dir = os.path.join(CONFIG_DIR, f'fold_{fold_num}')
        os.makedirs(fold_config_dir, exist_ok=True)
        
        train_txt_path = os.path.join(fold_config_dir, 'train.txt')
        val_txt_path = os.path.join(fold_config_dir, 'val.txt')
        
        with open(train_txt_path, 'w') as f:
            f.write('\n'.join(train_files))
        with open(val_txt_path, 'w') as f:
            f.write('\n'.join(val_files))
            
        # --- Create the new data_fold_k.yaml ---
        kfold_yaml_path = os.path.join(fold_config_dir, 'data_fold.yaml')
        kfold_yaml_data = {
            'train': train_txt_path,
            'val': val_txt_path,
            'nc': nc,
            'names': names
        }
        
        with open(kfold_yaml_path, 'w') as f:
            yaml.dump(kfold_yaml_data, f)
            
        # --- Train for this fold ---
        model = YOLO(base_model)
        model.to(device)
        
        results = model.train(
            data=kfold_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=lr0,
            lrf=lrf,
            device=device,
            project=kfold_project_dir,
            name=f'fold_{fold_num}'
        )
        
        final_map50 = results.box.map50
        all_metrics_map50.append(final_map50)
        print(f"Fold {fold_num} complete. Final mAP50: {final_map50:.4f}")

    # --- Print average results ---
    print(f"\n--- K-Fold Training Complete ---")
    mean_map50 = np.mean(all_metrics_map50)
    std_map50 = np.std(all_metrics_map50)
    
    print(f"All Fold mAP50 scores: {[round(m, 4) for m in all_metrics_map50]}")
    print(f"Average mAP50: {mean_map50:.4f} (+/- {std_map50:.4f})")
    
    # Return the path to the best model from the last fold
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    return best_model_path, mean_map50


def validate_on_test_set(model_path, test_set_files, nc, names, img_size, conf_threshold):
    """
    Validates the best model on the held-out test set.
    """
    print(f"\n--- Starting Final Validation on Hold-Out Test Set ---")
    print(f"Model: {model_path}")

    # --- Write test.txt ---
    test_config_dir = os.path.join(CONFIG_DIR, 'test_set')
    os.makedirs(test_config_dir, exist_ok=True)
    test_txt_path = os.path.join(test_config_dir, 'test.txt')
    
    with open(test_txt_path, 'w') as f:
        f.write('\n'.join(test_set_files))

    # --- Create test_data.yaml ---
    test_yaml_path = os.path.join(test_config_dir, 'test_data.yaml')
    test_yaml_data = {
        'train': test_txt_path,
        'val': test_txt_path,  # We point 'val' to our test.txt for model.val()
        'nc': nc,
        'names': names
    }
    with open(test_yaml_path, 'w') as f:
        yaml.dump(test_yaml_data, f)
        
    # --- Run Validation ---
    model = YOLO(model_path)
    metrics = model.val(
        data=test_yaml_path,
        imgsz=img_size,
        conf=conf_threshold,
        split='val'  # We use 'val' split because our YAML points 'val' to the test set
    )
    
    print("\n--- Final Test Set Metrics ---")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP75:     {metrics.box.map75:.4f}")


def predict_on_image(model_path, image_path, img_size, line_width=2):
    """
    Runs prediction on a single image and displays the result using matplotlib.
    """
    print(f"\n--- Running Prediction ---")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")

    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=img_size)

    if results:
        annotated_image_bgr = results[0].plot(line_width=line_width)
        annotated_image_rgb = annotated_image_bgr[..., ::-1] # BGR to RGB
        
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image_rgb)
        plt.title(f"Prediction result for: {os.path.basename(image_path)}")
        plt.axis("off")
        plt.show()

# --- 4. Main Execution ---

def main():
    """
    Main function to run the full pipeline.
    """
    
    # === Step 1: Split all data into a Train/Val pool and a Test set ===
    train_val_pool, test_set_files = prepare_data_splits(
        images_dir=ALL_IMAGES_DIR,
        labels_dir=ALL_LABELS_DIR,
        test_size=TEST_SET_SIZE
    )
    
    # === Step 2: Run K-Fold Training on the Train/Val pool ===
    best_model_from_kfold, avg_map50 = run_kfold_training(
        train_val_pool=train_val_pool,
        nc=NUM_CLASSES,
        names=CLASS_NAMES,
        base_model=BASE_MODEL,
        epochs=5,
        batch_size=16,
        img_size=640,
        lr0=0.0001,
        lrf=0.1,
        device=DEVICE,
        k=K_FOLDS
    )
    
    print(f"\nK-Fold process finished. Average mAP50: {avg_map50:.4f}")
    print(f"Best model from *last fold* saved at: {best_model_from_kfold}")
    
    # === Step 3: Validate the best model on the hold-out Test set ===
    validate_on_test_set(
        model_path=best_model_from_kfold,
        test_set_files=test_set_files,
        nc=NUM_CLASSES,
        names=CLASS_NAMES,
        img_size=640,
        conf_threshold=0.25
    )
    
    # === Step 4: Predict with the final model ===
    # (Select a random image from your test set for prediction)
    if test_set_files:
        image_to_test = test_set_files[0]
        predict_on_image(
            model_path=best_model_from_kfold,
            image_path=image_to_test,
            img_size=640
        )
    else:
        print("No test images to predict on.")


if __name__ == "__main__":
    main()