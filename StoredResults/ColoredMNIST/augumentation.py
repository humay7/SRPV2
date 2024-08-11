# augmentation.py
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def get_augmentation(augmentation_type):
    if augmentation_type == 'withoutda':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            #CenterPreservingBlur(kernel_size=21, sigma=10, preserve_ratio=0.5),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'blur':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'brightness':
        brightness_factor = 0.7
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=brightness_factor),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'colorjitter':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'contrast':
        contrast_factor = 0.7
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(contrast=contrast_factor),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'random_affine':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'rotation':
        angle=10
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=angle),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'saturation':
        saturation_factor = 0.5
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(saturation=saturation_factor),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'shear':
        shear = 10
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=0, shear=shear),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'random_transform':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),                  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'elastic_transform':
        return transforms.Compose([
         transforms.Resize((224, 224)),   
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    elif augmentation_type == 'random_invert':
        return transforms.Compose([
            transforms.Resize((224, 224)),
        transforms.RandomInvert(p=0.2),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    elif augmentation_type == 'random_posterize':
        return transforms.Compose([
            transforms.Resize((224, 224)),
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    elif augmentation_type == 'random_solarize':
        return transforms.Compose([
            transforms.Resize((224, 224)),
        transforms.RandomSolarize(threshold=150, p=0.3),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    elif augmentation_type == 'random_sharpeness':
        return transforms.Compose([
            transforms.Resize((224, 224)),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    elif augmentation_type == 'random_autocontrast':
        return transforms.Compose([
            transforms.Resize((224, 224)),
        transforms.RandomAutocontrast(p=0.4),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    elif augmentation_type == 'random_equalize':
        return transforms.Compose([
            transforms.Resize((224, 224)),
        transforms.RandomEqualize(p=0.4),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    else:
        raise ValueError("Invalid augmentation type")
