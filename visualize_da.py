import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import random
from torch.utils.data import Subset, DataLoader
import json
import threading
import csv
# %matplotlib inline
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# from /content/SRPV2/autoaugment import MNISTPolicy  # Import the MNIST-specific autoaugment policy
# %matplotlib inline
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import sys
import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)


class ImageNetPolicy(object):
   """ Randomly choose one of the best 24 Sub-policies on ImageNet.


       Example:
       >>> policy = ImageNetPolicy()
       >>> transformed = policy(image)


       Example as a PyTorch Transform:
       >>> transform=transforms.Compose([
       >>>     transforms.Resize(256),
       >>>     ImageNetPolicy(),
       >>>     transforms.ToTensor()])
   """
   def __init__(self, fillcolor=(128, 128, 128)):
       self.policies = [
           SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
           SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
           SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
           SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
           SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),


           SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
           SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
           SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
           SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
           SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),


           SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
           SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
           SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
           SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
           SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),


           SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
           SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
           SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
           SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
           SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),


           SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
           SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
           SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
           SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor)
       ]




   def __call__(self, img):
       policy_idx = random.randint(0, len(self.policies) - 1)
       return self.policies[policy_idx](img)


   def __repr__(self):
       return "AutoAugment ImageNet Policy"



class MNISTPolicy(object):
    """ Randomly choose one of the best 20 Sub-policies on MNIST.

        Example:
        >>> policy = MNISTPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     MNISTPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=128):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "autocontrast", 4, 0.8, "equalize", 6, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment MNIST Policy"



class SubPolicy(object):
   def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
       ranges = {
           "shearX": np.linspace(0, 0.3, 10),
           "shearY": np.linspace(0, 0.3, 10),
           "translateX": np.linspace(0, 150 / 331, 10),
           "translateY": np.linspace(0, 150 / 331, 10),
           "rotate": np.linspace(0, 30, 10),
           "color": np.linspace(0.0, 0.9, 10),
           "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
           "solarize": np.linspace(256, 0, 10),
           "contrast": np.linspace(0.0, 0.9, 10),
           "sharpness": np.linspace(0.0, 0.9, 10),
           "brightness": np.linspace(0.0, 0.9, 10),
           "autocontrast": [0] * 10,
           "equalize": [0] * 10,
           "invert": [0] * 10
       }


       # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
       def rotate_with_fill(img, magnitude):
           rot = img.convert("RGBA").rotate(magnitude)
           return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


       func = {
           "shearX": lambda img, magnitude: img.transform(
               img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
               Image.BICUBIC, fillcolor=fillcolor),
           "shearY": lambda img, magnitude: img.transform(
               img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
               Image.BICUBIC, fillcolor=fillcolor),
           "translateX": lambda img, magnitude: img.transform(
               img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
               fillcolor=fillcolor),
           "translateY": lambda img, magnitude: img.transform(
               img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
               fillcolor=fillcolor),
           "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
           # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
           "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
           "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
           "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
           "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
               1 + magnitude * random.choice([-1, 1])),
           "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
               1 + magnitude * random.choice([-1, 1])),
           "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
               1 + magnitude * random.choice([-1, 1])),
           "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
           "equalize": lambda img, magnitude: ImageOps.equalize(img),
           "invert": lambda img, magnitude: ImageOps.invert(img)
       }


       # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
       #     operation1, ranges[operation1][magnitude_idx1],
       #     operation2, ranges[operation2][magnitude_idx2])
       self.p1 = p1
       self.operation1 = func[operation1]
       self.magnitude1 = ranges[operation1][magnitude_idx1]
       self.p2 = p2
       self.operation2 = func[operation2]
       self.magnitude2 = ranges[operation2][magnitude_idx2]




   def __call__(self, img):
       if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
       if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
       return img










class CustomMNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, *args, apply_transform=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_transform = apply_transform
        self.transform_to_pil = transforms.Grayscale(num_output_channels=3)  # Transform to PIL
        self.tensor_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Return PIL image for augmentation
        if self.apply_transform:
            img = self.tensor_transform(img)  # Apply the tensor transformation
            target = torch.tensor(target)
        else:
            img = self.transform_to_pil(img)  # Keep the PIL format for augmentation
        return img, target

# Load the CustomMNISTDataset without applying any transforms (for the original image)
train_set = CustomMNISTDataset(root='/content/SRPV2/data', train=True, download=True, apply_transform=False)

# Set up the auto augment transform pipeline
auto_augment_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    MNISTPolicy(),  # AutoAugment policy for MNIST
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize for 3 channels
])

# Choose a single image from the dataset
img, label = train_set[0]  # Selecting the first image for example

# Apply the auto augment transformation
augmented_img = auto_augment_transform(img)

# Convert the tensors to numpy arrays for visualization
def tensor_to_image(tensor):
    tensor = tensor.clone().detach().cpu().numpy()
    tensor = tensor.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    tensor = tensor * 0.3081 + 0.1307  # Un-normalize using MNIST normalization values
    tensor = tensor.clip(0, 1)  # Clip values to keep them in the range [0, 1]
    return tensor

# Visualize the original and augmented images
original_img_np = tensor_to_image(transforms.ToTensor()(img))
augmented_img_np = tensor_to_image(augmented_img)

# Plotting the images side by side
plt.figure(figsize=(8, 4))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(original_img_np, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.tight_layout()
plt.show()

# Augmented image
plt.subplot(1, 2, 2)
plt.imshow(augmented_img_np, cmap='gray')
plt.title("Augmented Image")
plt.axis('off')

plt.show()
