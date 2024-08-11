import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

def color_grayscale_arr(arr, color='red'):
    """Converts grayscale image to either red, green, or blue"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])

    if color == 'red':
        arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    elif color == 'green':
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)
    elif color == 'blue':
        arr = np.concatenate([np.zeros((h, w, 2), dtype=dtype), arr], axis=2)
    else:
        raise ValueError(f"Unsupported color: {color}")

    return arr

class ColoredMNIST(datasets.VisionDataset):
    def __init__(self, root='./data', env='train', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.env = env  # Store the environment
        self.data_dir = os.path.join('ColoredMNIST', self.env)
        self.prepare_colored_mnist()
        self.data_label_tuples = []

    def __getitem__(self, index):
        img, target = self.data_label_tuples[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        dataset_dir = 'ColoredMNIST'
        os.makedirs(dataset_dir, exist_ok=True)

        colored_mnist_dir = os.path.join(dataset_dir, self.env)
        os.makedirs(colored_mnist_dir, exist_ok=True)

        # Create directories for labels
        for label in range(10):
            label_dir = os.path.join(colored_mnist_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)

        print(f'Preparing Colored MNIST for {self.env}')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
        test_mnist = datasets.mnist.MNIST(self.root, train=False, download=True)

        current_dataset = train_mnist if self.env == 'train' else test_mnist

        # Load the color mapping from the corresponding JSON file
        json_file = 'train.json' if self.env == 'train' else 'test.json'
        with open(json_file, 'r') as f:
            color_map = json.load(f)

        for idx, (im, label) in enumerate(current_dataset):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(current_dataset)}')
            im_array = np.array(im)

            color = color_map[str(label)]
            colored_arr = color_grayscale_arr(im_array, color=color)
            img = Image.fromarray(colored_arr)

            img_file_path = os.path.join(colored_mnist_dir, str(label), f'{idx}.png')
            img.save(img_file_path)



train_set = ColoredMNIST(root='./data', env='train')
test_set = ColoredMNIST(root='./data', env='test')
