import os
import numpy as np
from pathlib import Path
from skimage import io
import ray
from sklearn.preprocessing import FunctionTransformer

dir = Path('/Rice_Image_Dataset')

# load image dataset from directory
@ray.remote
def load_folder_image(folder):
    images = []
    labels = []
    folder_path = os.path.join(dir, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                img = io.imread(img_path)
                images.append(img)
                labels.append(folder)     
                
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
    
    
def load_image(dataset_path):
    ray.init(ignore_reinit_error=True, num_cpus=8)
    tasks = [load_folder_image.remote(folder) for folder in os.listdir(dataset_path)]
    
    all_results = ray.get(tasks)
    # Flatten list of images
    flat_results = [image for folder in all_results for image in folder]
    
    X, Y = zip(*flat_results)
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
    return X, Y

# load single image
def read_image_func(img_path):
    img = io.imread(img_path)
    return img

# transform function to integrate into data preprocessing pipeline
read_image = FunctionTransformer(read_image_func)
