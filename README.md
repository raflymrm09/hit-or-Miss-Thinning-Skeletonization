import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, skeletonize, thin
from skimage.util import invert
from google.colab import files  # Khusus untuk Google Colab

# Fungsi Hit-or-Miss
def hit_or_miss(image, se_foreground, se_background):
    image_complement = invert(image)
    eroded_foreground = erosion(image, se_foreground)
    eroded_background = erosion(image_complement, se_background)
    return eroded_foreground & eroded_background

# Upload gambar
uploaded = files.upload()
file_path = list(uploaded.keys())[0]

# Load gambar RGB
image_rgb = imread(file_path)

# Buang channel alpha jika ada
if image_rgb.shape[-1] == 4:
    image_rgb = image_rgb[:, :, :3]

# Konversi ke grayscale
image_gray = rgb2gray(image_rgb)

# Thresholding → binary image
thresh = threshold_otsu(image_gray)
image_binary = image_gray > thresh

# Structuring Element untuk Hit-or-Miss
se_foreground = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=bool)

se_background = np.array([
    [1, 0, 1],
    [0, 0, 0],
    [1, 0, 1]
], dtype=bool)

# Proses Hit-or-Miss
hitmiss_result = hit_or_miss(image_binary, se_foreground, se_background)

# Thinning
thinned = thin(image_binary)

# Skeletonization
skeleton = skeletonize(image_binary)

# Visualisasi hasil
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(image_gray, cmap='gray')
axs[0, 1].set_title("Grayscale")
axs[0, 1].axis('off')

axs[0, 2].imshow(image_binary, cmap='gray')
axs[0, 2].set_title("Binary")
axs[0, 2].axis('off')

axs[1, 0].imshow(hitmiss_result, cmap='gray')
axs[1, 0].set_title("Hit-or-Miss Result")
axs[1, 0].axis('off')

axs[1, 1].imshow(thinned, cmap='gray')
axs[1, 1].set_title("Thinning")
axs[1, 1].axis('off')

axs[1, 2].imshow(skeleton, cmap='gray')
axs[1, 2].set_title("Skeletonization")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()
