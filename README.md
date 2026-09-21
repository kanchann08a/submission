# submission
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
