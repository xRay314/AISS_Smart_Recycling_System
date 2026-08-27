import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

# =====================================================
# SETTINGS
# =====================================================
MODEL_FOLDER = Path("model3")
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / MODEL_FOLDER

EXPORT_FOLDER = MODEL_DIR / "export"
OUTPUT_FOLDER = MODEL_DIR / "dataset"

TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

MODEL = "yolo11n.pt"
EPOCHS = 100
IMAGE_SIZE = 640
BATCH = 16
WORKERS = 8
CACHE = True
DEVICE = 0
# =====================================================
# PATHS
# =====================================================

def main():

    export = Path(EXPORT_FOLDER)

    images = export / "images"
    labels = export / "labels"

    dataset = Path(OUTPUT_FOLDER)

    # =====================================================
    # CREATE DATASET FOLDERS
    # =====================================================

    if dataset.exists():
        shutil.rmtree(dataset)

    for split in ["train", "valid", "test"]:
        (dataset / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset / split / "labels").mkdir(parents=True, exist_ok=True)

    # =====================================================
    # FIND IMAGES
    # =====================================================

    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(images.glob(ext))

    print(f"Found {len(image_files)} images")

    # =====================================================
    # VERIFY LABELS
    # =====================================================

    valid_images = []

    for image in image_files:

        label = labels / (image.stem + ".txt")

        if label.exists():
            valid_images.append(image)
        else:
            print(f"Missing label: {image.name}")

    print(f"Using {len(valid_images)} labelled images")

    # =====================================================
    # SHUFFLE
    # =====================================================

    random.seed(42)
    random.shuffle(valid_images)

    # =====================================================
    # SPLIT
    # =====================================================

    total = len(valid_images)

    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train = valid_images[:train_end]
    valid = valid_images[train_end:valid_end]
    test = valid_images[valid_end:]

    print()

    print("Train:", len(train))
    print("Valid:", len(valid))
    print("Test :", len(test))

    # =====================================================
    # COPY FUNCTION
    # =====================================================

    def copy_files(files, split):

        for image in files:

            label = labels / (image.stem + ".txt")

            shutil.copy(
                image,
                dataset / split / "images" / image.name
            )

            shutil.copy(
                label,
                dataset / split / "labels" / label.name
            )

    copy_files(train, "train")
    copy_files(valid, "valid")
    copy_files(test, "test")

    # =====================================================
    # READ CLASSES
    # =====================================================

    classes = []

    with open(export / "classes.txt") as f:

        for line in f:

            line = line.strip()

            if line:
                classes.append(line)

    # =====================================================
    # CREATE data.yaml
    # =====================================================

    yaml = f"""path: "{dataset.resolve().as_posix()}"

train: train/images
val: valid/images
test: test/images

names:
"""

    for i, name in enumerate(classes):
        yaml += f"  {i}: {name}\n"

    DATA_YAML = MODEL_DIR / "data.yaml"

    with open(DATA_YAML, "w") as f:
        f.write(yaml)

    print("\ndata.yaml created")

    # =====================================================
    # TRAIN
    # =====================================================

    print("\nStarting training...\n")

    model = YOLO(MODEL)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        batch=BATCH,
        workers=WORKERS,
        cache=CACHE,
        project=str(MODEL_DIR / "runs"),
        name="train",
        exist_ok=True
    )

    print("\nTraining complete!")

    print("\nBest model:")

    print("runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
