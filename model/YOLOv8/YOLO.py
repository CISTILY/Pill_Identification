from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import warnings
import yaml
import glob
import numpy as np
from sklearn.model_selection import KFold, train_test_split # Added train_test_split
from hydra.core.hydra_config import HydraConfig

import hydra
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")

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


def run_kfold_training(train_val_pool, cfg, hydra_output_dir):
    """
    Runs K-fold cross-validation on the provided 'train_val_pool' of images.
    """
    print(f"\n--- Starting {cfg.training.K_FOLDS}-Fold Cross-Validation ---")
    
    kf = KFold(n_splits=cfg.training.K_FOLDS, shuffle=True, random_state=42)
    all_metrics_map50 = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(train_val_pool)):
        fold_num = fold_idx + 1
        print(f"\n--- Fold {fold_num}/{cfg.training.K_FOLDS} ---")
        
        # Get file paths for this fold
        train_files = [train_val_pool[i] for i in train_indices]
        val_files = [train_val_pool[i] for i in val_indices]
        
        # --- Write train.txt and val.txt for this fold ---
        fold_config_dir = os.path.join(cfg.paths.CONFIG_DIR, f'fold_{fold_num}')
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
            'nc': cfg.settings.NUM_CLASSES,
            'names': list(cfg.settings.CLASS_NAMES)
        }
        
        with open(kfold_yaml_path, 'w') as f:
            yaml.dump(kfold_yaml_data, f)
            
        # --- Train for this fold ---
        model = YOLO(cfg.paths.BASE_MODEL)
        model.to(cfg.settings.DEVICE)

        results = model.train(
            data=kfold_yaml_path,
            epochs=cfg.training.epochs,
            batch=cfg.training.batch_size,
            imgsz=cfg.training.img_size,
            lr0=cfg.training.lr0,
            lrf=cfg.training.lrf,
            device=cfg.settings.DEVICE,
            project=hydra_output_dir + "/yolo_runs",
            name=f'fold_{fold_num}',
            exist_ok=True
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


def validate_on_test_set(model_path, test_set_files, cfg, hydra_output_dir):
    """
    Validates the best model on the held-out test set.
    """
    print(f"\n--- Starting Final Validation on Hold-Out Test Set ---")
    print(f"Model: {model_path}")

    # --- Write test.txt ---
    test_config_dir = os.path.join(cfg.paths.CONFIG_DIR, 'test_set')
    os.makedirs(test_config_dir, exist_ok=True)
    test_txt_path = os.path.join(test_config_dir, 'test.txt')
    
    with open(test_txt_path, 'w') as f:
        f.write('\n'.join(test_set_files))

    # --- Create test_data.yaml ---
    test_yaml_path = os.path.join(test_config_dir, 'test_data.yaml')
    test_yaml_data = {
        'train': test_txt_path,
        'val': test_txt_path,  # We point 'val' to our test.txt for model.val()
        'nc': cfg.settings.NUM_CLASSES,
        'names': list(cfg.settings.CLASS_NAMES)
    }
    with open(test_yaml_path, 'w') as f:
        yaml.dump(test_yaml_data, f)
        
    # --- Run Validation ---
    model = YOLO(model_path)
    metrics = model.val(
        data=test_yaml_path,
        imgsz=cfg.training.img_size,
        conf=cfg.settings.CONF_THRESHOLD,
        project=hydra_output_dir,
        split='val'  # We use 'val' split because our YAML points 'val' to the test set
    )
    
    print("\n--- Final Test Set Metrics ---")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP75:     {metrics.box.map75:.4f}")


def predict_on_image(model_path, image_path, cfg, line_width=2):
    """
    Runs prediction on a single image and displays the result using matplotlib.
    """
    print(f"\n--- Running Prediction ---")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")

    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=cfg.training.img_size)

    if results:
        annotated_image_bgr = results[0].plot(line_width=line_width)
        annotated_image_rgb = annotated_image_bgr[..., ::-1] # BGR to RGB
        
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image_rgb)
        plt.title(f"Prediction result for: {os.path.basename(image_path)}")
        plt.axis("off")
        plt.show()

# --- 4. Main Execution ---
@hydra.main(config_path="Configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the full pipeline.
    """
    
    print(OmegaConf.to_yaml(cfg))

    os.makedirs(cfg.paths.CONFIG_DIR, exist_ok=True)

    print(f"▶ Running with LR={cfg.training.lr0}, "
          f"BATCH_SIZE={cfg.training.batch_size}, "
          f"EPOCHS={cfg.training.epochs}, "
          f"K={cfg.training.K_FOLDS}")
    
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    print("Hydra output dir:", hydra_output_dir)

    # === Step 1: Split all data into a Train/Val pool and a Test set ===
    train_val_pool, test_set_files = prepare_data_splits(
        images_dir=cfg.paths.ALL_IMAGES_DIR,
        labels_dir=cfg.paths.ALL_LABELS_DIR,
        test_size=cfg.training.TEST_SET_SIZE
    )
    
    # === Step 2: Run K-Fold Training on the Train/Val pool ===
    best_model_from_kfold, avg_map50 = run_kfold_training(
        train_val_pool=train_val_pool,
        cfg=cfg,
        hydra_output_dir=hydra_output_dir
    )
    
    print(f"\nK-Fold process finished. Average mAP50: {avg_map50:.4f}")
    print(f"Best model from *last fold* saved at: {best_model_from_kfold}")
    
    # === Step 3: Validate the best model on the hold-out Test set ===
    validate_on_test_set(
        model_path=best_model_from_kfold,
        test_set_files=test_set_files,
        cfg=cfg,
        hydra_output_dir=hydra_output_dir
    )
    
    # === Step 4: Predict with the final model ===
    # (Select a random image from your test set for prediction)
    if test_set_files:
        image_to_test = test_set_files[0]
        predict_on_image(
            model_path=best_model_from_kfold,
            image_path=image_to_test,
            cfg=cfg
        )
    else:
        print("No test images to predict on.")


if __name__ == "__main__":
    main()