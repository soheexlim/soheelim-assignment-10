import os
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Grayscale
            img = img.resize(target_size)
            img_array = np.asarray(img, dtype=np.float32) / 255.0
            images.append(img_array.flatten())
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names

# Load images
image_dir = "/path/to/images"
train_images, train_image_names = load_images(image_dir, max_images=2000)

# PCA
k = 50
pca = PCA(n_components=k)
pca.fit(train_images)
