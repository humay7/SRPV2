import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_images_from_directory(directory, num_images=2):
    """Load a specified number of random images from a directory."""
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    images = []
    for file in selected_files:
        img_path = os.path.join(directory, file)
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB mode
        images.append(img)
    return images

def plot_images_grid(root_dir, num_images_per_label=2):
    """Plot a grid of images with 2 rows and 10 columns from each label directory."""
    labels = sorted(os.listdir(root_dir))  # Assuming labels are directory names
    num_labels = len(labels)
    
    if num_labels != 10:
        raise ValueError("Expected exactly 10 label directories for proper grid layout.")
    
    fig, axes = plt.subplots(nrows=2, ncols=num_labels, figsize=(20, 5))
    
    for i, label in enumerate(labels):
        label_dir = os.path.join(root_dir, label)
        images = load_images_from_directory(label_dir, num_images=num_images_per_label)
        
        for j, img in enumerate(images):
            ax = axes[j, i]  # Adjust axis position
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(f'Label: {label}')
    
    plt.tight_layout()
    plt.show()

# Define the root directory for training images

inp = input('Enter train or test: ')

if inp=='train':
    root_dir = './ColoredMNIST/train' 
elif inp=='test':
    root_dir = './ColoredMNIST/test' 


plot_images_grid(root_dir, num_images_per_label=2)
