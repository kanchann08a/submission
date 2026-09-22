from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "testt" / "eval_images"
TEST_METADATA = BASE_DIR / "testt" / "test_metadata.csv"
CHECKPOINT = BASE_DIR / "checkpoints" / "best_exp4_extended_cosine.pth"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "submission.csv"

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("LUNAR SURFACE CLASSIFICATION - INFERENCE")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Test directory: {TEST_DIR}")
print(f"Test metadata: {TEST_METADATA}")
print(f"Checkpoint: {CHECKPOINT}")
print()


# ============================================================
# CHECK FILES
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )

if not TEST_METADATA.exists():
    raise FileNotFoundError(
        f"Test metadata not found: {TEST_METADATA}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PHYSICS-BASED ROTATION
# ============================================================

def rotate_by_sun_azimuth(image, sun_azimuth_angle):
    """
    Rotate image counter-clockwise by -sun_azimuth_angle.

    Bicubic interpolation is used.
    Empty regions are filled using the image mean.
    """

    # Convert to grayscale
    image = image.convert("L")

    # Calculate mean intensity
    image_array = np.asarray(image, dtype=np.float32)
    mean_intensity = float(image_array.mean())

    # Rotate by negative sun azimuth
    rotated = image.rotate(
        -float(sun_azimuth_angle),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=int(round(mean_intensity))
    )

    return rotated


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    # Same ImageNet normalization used for ResNet18
    transforms.Normalize(
        mean=[0.485, 0.485, 0.485],
        std=[0.229, 0.229, 0.229]
    )
])


# ============================================================
# TEST DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(
        self,
        metadata,
        image_dir,
        transform=None
    ):

        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # ----------------------------------------------------
        # Build image lookup
        # ----------------------------------------------------

        self.image_lookup = {}

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG"
        }

        for image_path in self.image_dir.iterdir():

            if image_path.is_file() and image_path.suffix in extensions:

                # Store using filename stem
                self.image_lookup[image_path.stem] = image_path

        print(
            f"Images found in test directory: "
            f"{len(self.image_lookup)}"
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        image_id = str(row["image_id"])
        sun_azimuth = float(row["sun_azimuth_angle"])

        # ----------------------------------------------------
        # Find image
        # ----------------------------------------------------

        image_path = None

        # Case 1:
        # image_id already contains extension
        #
        # Example:
        # eval_00001.png
        #
        # Do NOT create:
        # eval_00001.png.png
        # ----------------------------------------------------

        image_id_path = Path(image_id)

        if image_id_path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png"
        }:

            candidate = self.image_dir / image_id

            if candidate.exists():
                image_path = candidate

        # ----------------------------------------------------
        # Case 2:
        # image_id does not contain extension
        # ----------------------------------------------------

        if image_path is None:

            possible_names = [
                f"{image_id}.jpg",
                f"{image_id}.jpeg",
                f"{image_id}.png",
                f"{image_id}.JPG",
                f"{image_id}.JPEG",
                f"{image_id}.PNG"
            ]

            for name in possible_names:

                candidate = self.image_dir / name

                if candidate.exists():

                    image_path = candidate
                    break

        # ----------------------------------------------------
        # Case 3:
        # Fallback using filename stem
        # ----------------------------------------------------

        if image_path is None:

            image_stem = Path(image_id).stem

            image_path = self.image_lookup.get(
                image_stem
            )

        # ----------------------------------------------------
        # Image still not found
        # ----------------------------------------------------

        if image_path is None:

            raise FileNotFoundError(
                f"Could not find image for image_id: "
                f"{image_id}"
            )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("L")

        # ----------------------------------------------------
        # Physics preprocessing
        # ----------------------------------------------------

        image = rotate_by_sun_azimuth(
            image,
            sun_azimuth
        )

        # ----------------------------------------------------
        # Convert grayscale to RGB
        # ----------------------------------------------------
        # ResNet18 expects 3 channels.
        # ----------------------------------------------------

        image = image.convert("RGB")

        # ----------------------------------------------------
        # Apply transform
        # ----------------------------------------------------

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id


# ============================================================
# LOAD TEST METADATA
# ============================================================

print("Loading test metadata...")

test_metadata = pd.read_csv(TEST_METADATA)

print(f"Test metadata rows: {len(test_metadata)}")

print("\nMetadata columns:")
print(test_metadata.columns.tolist())

# Required columns
required_columns = {
    "image_id",
    "sun_azimuth_angle"
}

missing_columns = required_columns - set(
    test_metadata.columns
)

if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# CREATE DATASET
# ============================================================

test_dataset = TestDataset(
    metadata=test_metadata,
    image_dir=TEST_DIR,
    transform=transform
)

print(
    f"\nDataset size: {len(test_dataset)}"
)

if len(test_dataset) != 2000:

    print(
        f"WARNING: Expected 2000 test images, "
        f"but found {len(test_dataset)} metadata rows."
    )


# ============================================================
# CREATE DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Number of batches: {len(test_loader)}"
)

print()


# ============================================================
# CREATE RESNET18
# ============================================================

print("Loading ResNet18 model...")

model = models.resnet18(
    weights=None
)

# Two classes:
# 0 = Depth
# 1 = Rise

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

# ------------------------------------------------------------
# Handle different checkpoint formats
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ------------------------------------------------------------
# Remove "module." prefix if model was saved using DataParallel
# ------------------------------------------------------------

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[len("module."):]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict
)

model.to(DEVICE)

model.eval()

print("Checkpoint loaded successfully.")

print()


# ============================================================
# INFERENCE
# ============================================================

print("=" * 60)
print("RUNNING INFERENCE")
print("=" * 60)

test_image_ids = []
test_predictions = []

total_batches = len(test_loader)

with torch.inference_mode():

    for batch_idx, (inputs, image_ids) in enumerate(
        test_loader
    ):

        # Move images to device
        inputs = inputs.to(DEVICE)

        # Model prediction
        outputs = model(inputs)

        # Get predicted class
        predictions = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

        # Store
        test_image_ids.extend(
            image_ids
        )

        test_predictions.extend(
            predictions.tolist()
        )

        # Progress
        print(
            f"Batch {batch_idx + 1}/{total_batches}",
            end="\r"
        )


print()
print()
print("Inference completed.")
print(
    f"Predictions generated: "
    f"{len(test_predictions)}"
)

print()


# ============================================================
# CREATE SUBMISSION DATAFRAME
# ============================================================

sub_df = pd.DataFrame({
    "image_id": test_image_ids,
    "label": test_predictions
})


# ============================================================
# VALIDATE SUBMISSION
# ============================================================

print("=" * 60)
print("VALIDATING SUBMISSION")
print("=" * 60)

# Check number of rows
if len(sub_df) != 2000:

    raise ValueError(
        f"Submission should contain 2000 rows, "
        f"but contains {len(sub_df)}"
    )

print("✓ Row count: 2000")


# Check columns
expected_columns = [
    "image_id",
    "label"
]

if list(sub_df.columns) != expected_columns:

    raise ValueError(
        f"Incorrect columns: "
        f"{sub_df.columns.tolist()}"
    )

print("✓ Columns: image_id, label")


# Check duplicate IDs
duplicate_count = sub_df["image_id"].duplicated().sum()

if duplicate_count > 0:

    raise ValueError(
        f"Found {duplicate_count} duplicate image IDs"
    )

print("✓ No duplicate image IDs")


# Check missing values
if sub_df.isnull().any().any():

    raise ValueError(
        "Submission contains missing values"
    )

print("✓ No missing values")


# Check labels
unique_labels = set(
    sub_df["label"].unique()
)

if not unique_labels.issubset({0, 1}):

    raise ValueError(
        f"Invalid labels found: {unique_labels}"
    )

print("✓ Labels contain only 0 and 1")


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nPrediction distribution:")

print(
    sub_df["label"].value_counts().sort_index()
)

print()

print(
    "Class 0 = Depth"
)

print(
    "Class 1 = Rise"
)

print()


# ============================================================
# SAVE SUBMISSION
# ============================================================

sub_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# Also save a copy in project root
root_submission = BASE_DIR / "submission.csv"

sub_df.to_csv(
    root_submission,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("=" * 60)
print("SUBMISSION CREATED SUCCESSFULLY")
print("=" * 60)

print(
    f"Output file:\n{OUTPUT_FILE}"
)

print(
    f"Root copy:\n{root_submission}"
)

print()

print("First 10 predictions:")
print(
    sub_df.head(10).to_string(index=False)
)

print()

print("Last 10 predictions:")
print(
    sub_df.tail(10).to_string(index=False)
)

print()

print("=" * 60)
print("DONE")
print("=" * 60)