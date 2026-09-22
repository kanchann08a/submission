Balanced Accuracy: 73.99%
## How to Run `inference.py`

`inference.py` generates predictions for all 2,000 evaluation images and creates the final `submission.csv`.

### 1. Open Terminal

Go to the project folder:

```bash
cd /Users/kanchannishad/Desktop/submission
2. Run the inference script

Use the Python environment containing PyTorch, NumPy, Pandas, Pillow, and Torchvision:

/opt/anaconda3/bin/python inference.py
3. Required files

Before running, make sure these files/folders exist:

submission/
├── inference.py
├── checkpoints/
│   └── best_exp4_extended_cosine.pth
└── testt/
    ├── test_metadata.csv
    └── eval_images/
        ├── eval_00001.png
        ├── eval_00002.png
        └── ...
```
# The Pareidolia Paradox — Lunar Surface Classification

Lunar surface classification using physics-aware image preprocessing and a ResNet18 deep learning model.

The task is to classify lunar surface images into two classes:

- **0 — Depth:** craters, holes, depressions
- **1 — Rise:** mounds, hills, rocks, boulders

---

## 1. Project Overview

This project classifies lunar terrain images as either a **Depth** or a **Rise**.

The dataset provides:

- Lunar surface images
- Sun azimuth angle for each image
- Training labels for the training set

The sun azimuth angle is used during preprocessing to normalize the illumination direction before classification.

The final model is a **ResNet18** trained for binary classification.

### Final Validation Result

**Validation Balanced Accuracy: 73.99%**

- Train/Validation split: **85/15**
- Stratified split
- `random_state = 42`
- Best checkpoint: **Epoch 14**

---

# 2. Project Structure

```text
submission/
│
├── README.md
├── requirements.txt
│
├── paradox.ipynb
│
├── inference.py
├── run_exp4.py
├── run_exp5.py
│
├── checkpoints/
│   └── best_exp4_extended_cosine.pth
│
├── train/
│   ├── train_metadata.csv
│   └── training images
│
├── testt/
│   ├── test_metadata.csv
│   │
│   ├── eval_images/
│   │   └── evaluation images
│   │
│   └── eval_images_rot/
│       └── rotated evaluation images
│
├── outputs/
│   └── submission.csv
│
└── submission.csv
```

# Technical Approach

## 1. Core Idea

The key observation behind the approach is that lunar terrain classification is strongly influenced by **illumination geometry**.

The model is not simply learning whether an image "looks like" a crater or a rock. Lunar surface structures produce characteristic patterns of **shadows, highlights, edges, and local intensity variations** depending on the direction from which sunlight illuminates the terrain.

Therefore, the main approach was to explicitly use the provided **Sun Azimuth Angle** to normalize the orientation of the images before performing deep-learning-based classification.

The complete pipeline was:

Sun Azimuth
      ↓
Physics-aware orientation normalization
      ↓
Grayscale image
      ↓
Spatial transformation
      ↓
ResNet18 feature extraction
      ↓
Binary classification
      ↓
Depth / Rise

---

# 2. Physics-Aware Preprocessing

## Why Sun Azimuth Matters

The appearance of a lunar surface feature depends heavily on the direction of illumination.

For example, consider a crater:

- The crater geometry remains the same.
- However, changing the direction of sunlight changes which side of the crater is illuminated.
- The opposite side can become darker due to shadowing.
- Consequently, the same physical structure can produce different pixel patterns.

The same effect occurs for raised structures such as rocks, hills, and mounds.

This creates an important distinction between:

**Geometric structure**

and

**illumination-dependent appearance**.

Instead of allowing the neural network to learn all possible illumination orientations independently, the preprocessing stage attempts to normalize this variation.

---

# 3. Sun Azimuth Normalization

For each image, the provided Sun Azimuth Angle was used to rotate the image by:

```text
Rotation = -Sun Azimuth Angle
````

The rotation was performed counter-clockwise using bicubic interpolation.

Conceptually:

```text
Original Image
      +
Sun Azimuth Angle
      ↓
Orientation Normalization
      ↓
Standardized Illumination Orientation
```

This makes the visual representation of terrain structures more comparable across samples.

### Boundary Handling

Image rotation creates regions outside the original image boundaries.

Instead of introducing artificial black pixels, the empty regions were filled using the **mean intensity of the image**.

This avoids creating strong artificial edges that could otherwise become features learned by the CNN.

---

# 4. Grayscale Representation

The images were converted to grayscale before being passed to the classification pipeline.

The task is primarily driven by geometric and photometric information such as:

* Surface texture
* Intensity gradients
* Shadow regions
* Crater boundaries
* Rock boundaries
* Local contrast
* Raised/depressed structures

Color is therefore not the primary source of information for this problem.

Using grayscale also focuses the model on the intensity structure produced by the interaction between terrain geometry and illumination.

---

# 5. Data Split

The labeled data was divided using an **85:15 stratified split**.

```text
Training   : 6,675 images
Validation : 1,179 images
Random State: 42
```

Stratification was important because the validation set needed to preserve the class distribution.

The evaluation images were kept separate and were not used for model selection.

---

# 6. Model Selection

The final architecture used was **ResNet18**.

ResNet18 was useful because the task depends on learning hierarchical spatial features rather than simple pixel-level differences.

The CNN can progressively learn features such as:

```text
Pixels
  ↓
Edges and gradients
  ↓
Textures and local structures
  ↓
Crater / rock boundaries
  ↓
Higher-level terrain patterns
  ↓
Depth vs Rise
```

## Why ResNet18?

ResNet18 provides a relatively compact convolutional architecture while still providing strong hierarchical feature extraction.

Its residual connections allow information to pass through the network using shortcut connections:

```text
Input
  ├───────────────┐
  ↓               │
Convolution       │
  ↓               │
Convolution       │
  ↓               │
  + ←─────────────┘
  ↓
Output
```

This makes the network easier to optimize than a similarly deep plain CNN.

For this problem, ResNet18 provided a useful balance between:

* Spatial feature extraction
* Model complexity
* Training cost
* Generalization

---

# 7. Classification Head

The original ResNet18 classification layer was replaced with a two-class output layer.

```text
ResNet18 Backbone
        ↓
Learned Feature Representation
        ↓
Fully Connected Layer
        ↓
2 Outputs
   ↙       ↘
Depth     Rise
```

The two output classes were:

```text
0 → Depth
1 → Rise
```

The model therefore learns a decision boundary in the learned feature space separating depressed terrain from raised terrain.

---

# 8. Why Balanced Accuracy Was Used

The primary evaluation metric was **Balanced Accuracy**, rather than relying only on conventional accuracy.

Balanced Accuracy is:

```text
Balanced Accuracy =
(Recall_Depth + Recall_Rise) / 2
```

where:

```text
Recall_Depth = TP_Depth / (TP_Depth + FN_Depth)

Recall_Rise = TP_Rise / (TP_Rise + FN_Rise)
```

This metric is particularly useful for a two-class problem because it gives equal importance to both classes.

A model that performs very well on one class but poorly on the other should not appear artificially strong simply because of the class distribution.

---

# 9. Validation Results

The best validation result was obtained at:

```text
Epoch: 14
Balanced Accuracy: 73.99%
```

The Epoch 14 checkpoint was therefore selected as the final model.

```text
checkpoints/best_exp4_extended_cosine.pth
```

Importantly, the **best checkpoint was selected using validation performance**, rather than automatically using the final training epoch.

Later training showed a reduction in validation Balanced Accuracy, indicating that continuing training was not improving generalization.

Therefore:

```text
Best validation checkpoint
        ↓
Epoch 14
        ↓
73.99% Balanced Accuracy
```

was used for final inference.

---

# 10. Confusion Matrix Analysis

The final validation confusion matrix was:

```text
                    Predicted
                  Depth    Rise

Actual Depth       329      99

Actual Rise        217     534
```

Therefore:

### Depth

```text
Correctly classified: 329
Misclassified as Rise: 99
```

### Rise

```text
Correctly classified: 534
Misclassified as Depth: 217
```

The confusion matrix provides information that a single accuracy value cannot provide.

In particular, it shows that the model has different error behavior for the two terrain categories.

The relatively large number of Rise → Depth errors indicates that some raised structures can produce visual patterns that resemble depressions after illumination and shadow formation.

This is physically plausible because a bright/dark boundary can arise from either:

* A concave structure such as a crater
* A convex structure such as a rock or mound

depending on the illumination direction.

This is one reason the Sun Azimuth information is important for the problem.

---

