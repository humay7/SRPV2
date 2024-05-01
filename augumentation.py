# augmentation.py
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

def get_augmentation(augmentation_type):
    if augmentation_type == 'al_withoutda':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'blur':
        return transforms.Compose([
            transforms.GaussianBlur(kernel_size=3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'brightness':
        brightness_factor = 0.7
        return transforms.Compose([
            transforms.ColorJitter(brightness=brightness_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'colorjitter':
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'contrast':
        contrast_factor = 0.7
        return transforms.Compose([
            transforms.ColorJitter(contrast=contrast_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'histogram_equalization':
        class HistogramEqualization(object):
            def __call__(self, img):
                np_img = np.array(img)
                eq_img = np.zeros_like(np_img)
                for c in range(3):
                    eq_img[:, :, c] = cv2.equalizeHist(np_img[:, :, c])
                return Image.fromarray(eq_img)

        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            HistogramEqualization(),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'random_affine':
        return transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'rotation':
        angle=10
        return transforms.Compose([
            transforms.RandomRotation(degrees=angle),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'saturation':
        saturation_factor = 0.5
        return transforms.Compose([
            transforms.ColorJitter(saturation=saturation_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'shear':
        shear = 10
        return transforms.Compose([
            transforms.RandomAffine(degrees=0, shear=shear),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'single_channel':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        raise ValueError("Invalid augmentation type")
