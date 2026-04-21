import os
import shutil
import random

SOURCE_DIR  = "backend\\data\\vision\\raw"  
OUTPUT_DIR  = "backend\\data\\vision\\dataset"           
VAL_SPLIT   = 0.2                    
RANDOM_SEED = 42                      

def split_dataset():
    random.seed(RANDOM_SEED)

    train_dir = os.path.join(OUTPUT_DIR, "train")
    val_dir   = os.path.join(OUTPUT_DIR, "val")

    # Check source exists
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

    # Check output doesn't already exist (prevent accidental re-run)
    if os.path.exists(OUTPUT_DIR):
        print(f"WARNING: '{OUTPUT_DIR}' already exists.")
        answer = input("Do you want to delete it and re-split? (yes/no): ").strip().lower()
        if answer != "yes":
            print("Aborted. Your existing split is unchanged.")
            return
        shutil.rmtree(OUTPUT_DIR)
        print(f"Deleted existing '{OUTPUT_DIR}'.")

    class_names = [
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ]

    if not class_names:
        raise ValueError(f"No subfolders found in {SOURCE_DIR}")

    print(f"\nFound {len(class_names)} classes: {class_names}\n")

    total_train = 0
    total_val   = 0

    for class_name in sorted(class_names):
        class_path = os.path.join(SOURCE_DIR, class_name)

        # Get all image files
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if not images:
            print(f"  SKIPPING {class_name} — no images found")
            continue

        random.shuffle(images)

        split_idx  = max(1, int((1 - VAL_SPLIT) * len(images)))
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        # Create output folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir,   class_name), exist_ok=True)

        # Copy images
        for img in train_imgs:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(train_dir, class_name, img)
            )
        for img in val_imgs:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(val_dir, class_name, img)
            )

        total_train += len(train_imgs)
        total_val   += len(val_imgs)

        print(f"  {class_name:25s}  train={len(train_imgs):4d}  val={len(val_imgs):4d}")

    print(f"\nDone!")
    print(f"  Total training images:   {total_train}")
    print(f"  Total validation images: {total_val}")
    print(f"\nFolders created:")
    print(f"  {train_dir}/")
    print(f"  {val_dir}/")
    print("\nDo NOT run this file again unless you want to redo the split.")


if __name__ == "__main__":
    split_dataset()