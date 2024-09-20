from PIL.Image import new
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
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import ConcatDataset
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
import sys
sys.path.append('/content/SRPV2/StoredResults/ColoredMNIST/autoaugment.py')

with open('/content/SRPV2/StoredResults/ColoredMNIST/parameters.json', 'r') as f:
   params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
images_per_class = params['images_per_class']
num_runs = params['num_runs']
n_samples_add_pool = params['n_samples_add_pool']

# from autoaugment import ImageNetPolicy  # Import or define ImageNetPolicy
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


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




# if __name__ == '__main__':
#    from matplotlib import pyplot as plt




#    def rotate_with_fill(img, magnitude):
#        rot = img.convert("RGBA").rotate(magnitude)
#        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


#    dataset_root = "/data/DataSets/miniImageNet"


#    img_paths = ['n0153282900000018.jpg',
#                 'n0153282900000019.jpg',
#                 'n0153282900000020.jpg',
#                 'n0153282900000021.jpg',
#                 'n0153282900000023.jpg']


#    img_paths = [os.path.join(dataset_root, 'images', item) for item in img_paths]
#    img = Image.open(img_paths[0]).convert('RGB')


#    changed_img = rotate_with_fill(img, 0)


#    plt.figure()
#    plt.subplot(121)
#    plt.imshow(img)
#    plt.subplot(122)
#    plt.imshow(changed_img)
#    plt.show()










transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

auto_augment_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    ImageNetPolicy(),  # AutoAugment policy for ImageNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class AutoAugmentSubset(Subset):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = auto_augment_transform(img)  # Apply auto augment transform here
        return img, label

# Define rotation transforms for the unlabeled set
rotation_transforms = []
for i in range(4):
    angle = i * 90
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # Apply cropping with some padding
        transforms.ColorJitter(brightness=0.4, contrast=0.4),  # Adjust brightness and contrast
        transforms.RandomRotation(degrees=angle),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rotation_transforms.append(transform)

# Function to randomly apply one of the rotation transformations
def apply_random_transform(img):
    transform = random.choice(rotation_transforms)  # Select a random transformation
    return transform(img)

# CustomMNISTDataset for switching between PIL and Tensor formats
class CustomMNISTDataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, apply_transform=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_transform = apply_transform
        self.transform_to_pil = transforms.Resize((224, 224))  # Transform to PIL
        self.tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Return PIL image for augmentation
        if self.apply_transform:
            img = self.tensor_transform(img)  # Apply the tensor transformation
        else:
            img = self.transform_to_pil(img)  # Keep the PIL format for augmentation
        return img, target

from torchvision import transforms

# Define a new transformation that converts labels to tensors
class TensorLabelSubset(Subset):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, torch.tensor(label)  # Convert label to tensor

# Load datasets
train_set = CustomMNISTDataset(root='/content/SRPV2/StoredResults/ColoredMNIST/ColoredMNIST/train', apply_transform=False)
test_set = datasets.ImageFolder(root='/content/SRPV2/StoredResults/ColoredMNIST/ColoredMNIST/test', transform=transform_test)
    

# Print dataset stats
print('Dataset Stats:')
print(f'Train Dataset Size: {len(train_set)}, Test Dataset Size: {len(test_set)}\n')
def uncertainty_sampling(model, unlabeled_loader, n_samples):
    def entropy(p):
        return -torch.sum(p * torch.log2(p), dim=1)

    uncertainties = []
    with torch.no_grad():
        for data in unlabeled_loader:
            images, _ = data
            images = images.cuda()
            outputs = torch.softmax(model(images), dim=1)
            uncertainties.extend(entropy(outputs).tolist())
  
    # Select indices of top uncertain samples
    top_indices = np.argsort(uncertainties)[-n_samples:]
    return top_indices

# Define a function to perform a single run
def single_run(run_number):
    # Initialize a new model for each run
    model = torchvision.models.resnet18(pretrained=False)  # ResNet18 expects 3-channel input
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Modify the last fully connected layer for 10 classes (MNIST)
    model.cuda()  # Move model to GPU

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Split train set into labeled and unlabeled indices


# Assuming train_set and test_set are defined elsewhere
# images_per_class: Number of images per class for the labeled set
# batch_size: Your defined batch size for DataLoader
# epochs: Number of training epochs
# run_number: Run identifier
# n_samples_add_pool: Number of samples to add from uncertainty sampling
# model, optimizer, criterion: Your model and optimizer

# Initializing empty lists for labeled and unlabeled indices
    # Split the dataset into labeled and unlabeled indices
    labeled_indices = []
    unlabeled_indices = []

    # Loop over each class and randomly select 'images_per_class' for labeled set
    for class_label in range(10):  # Assuming 10 classes for example
        class_indices = np.where(np.array(train_set.targets) == class_label)[0]
        np.random.shuffle(class_indices)  # Shuffle class-specific indices
        selected_indices = class_indices[:min(images_per_class, len(class_indices))]
        labeled_indices.extend(selected_indices)

    # Create labeled and unlabeled sets from the original dataset
    labeled_set = Subset(train_set, labeled_indices)
    unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices))
    unlabeled_set = TensorLabelSubset(train_set, unlabeled_indices)  # Assuming you have TensorLabelSubset class

    # Augmentation process for labeled set
    augmented_images = []
    augmented_labels = []
    augmented_indices = []

    # Start index for augmented images (ensure no conflict with existing indices)
    start_idx = len(train_set)  # Start after the end of original train_set

    # Set the train_set to return PIL images for augmentation
    train_set.apply_transform = False  # Ensure it returns PIL images for augmentation

    # Apply augmentation to the labeled set
    for img_idx, (img, label) in enumerate(labeled_set):
        augmented_image = apply_random_transform(img)  # Apply the random transformation
        augmented_images.append(augmented_image)  # Append transformed image (tensor)
        augmented_labels.append(torch.tensor(label))  # Ensure label is a tensor
        augmented_indices.append(start_idx + len(augmented_images) - 1)  # Assign new unique index

    # Create augmented dataset from augmented images and labels
    augmented_dataset = torch.utils.data.TensorDataset(torch.stack(augmented_images), torch.tensor(augmented_labels))

    # Combine the original unlabeled set with the augmented dataset
    combined_unlabeled_set = torch.utils.data.ConcatDataset([unlabeled_set, augmented_dataset])

    # Update the unlabeled indices with the new augmented indices
    unlabeled_indices = list(set(range(len(unlabeled_set)))) + augmented_indices  # Only use valid indices for unlabeled_set and add augmented indices

    # Update DataLoaders for the new combined dataset
    train_set.apply_transform = True  # Switch back to Tensor format for DataLoader

    # Concatenate the labeled_set and combined_unlabeled_set into new_train_set
    new_train_set = torch.utils.data.ConcatDataset([labeled_set, combined_unlabeled_set])

    # Adjusting labeled_indices and unlabeled_indices to reflect the new concatenated dataset
    len_labeled = len(labeled_set)
    labeled_indices = [i for i in range(len_labeled)]  # Keep original indices for labeled_set
    unlabeled_indices = [i + len_labeled for i in range(len(unlabeled_set))]  # Offset by labeled_set length for unlabeled_set

    # Update DataLoaders
    labeled_loader = DataLoader(Subset(new_train_set, labeled_indices), batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(Subset(new_train_set, unlabeled_indices), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # At this point, new_train_set contains both labeled and unlabeled data,
    # and DataLoaders are updated accordingly

    # new_train_set = torch.utils.data.ConcatDataset([labeled_set, combined_unlabeled_set])
    print(len(labeled_set))
    print(len(new_train_set))

     # Training loop
    for epoch in range(1, epochs + 1):
        # Train the model
        train_loss, train_accuracy = train_model(model, labeled_loader, optimizer, criterion)  # Assuming train_model function exists
        # test_accuracy = test_model(model, test_loader)  # Assuming test_model function exists
        print(f'Run {run_number} -> Epoch [{epoch}/{epochs}], LP:{len(labeled_indices)}, UP:{len(unlabeled_indices)}, Train Acc: {train_accuracy:.2f}%, Loss: {train_loss:.6f}')

        # Store results (Assuming results_writer is defined)
        # results_writer.writerow([run_number, epoch, train_loss, train_accuracy, test_accuracy])

        if epoch < epochs:  # Avoid running uncertainty sampling in the last iteration
            uncertain_indices = uncertainty_sampling(model, unlabeled_loader, n_samples_add_pool)  # Assuming uncertainty_sampling function exists
            
            # Extend labeled_indices with new uncertain samples
            labeled_indices.extend(uncertain_indices)
            
            # Update unlabeled_indices after removing the newly labeled ones
            unlabeled_indices = list(set(range(len(new_train_set))) - set(labeled_indices))

            # Update loaders with the new labeled and unlabeled sets
            labeled_loader = DataLoader(Subset(new_train_set, labeled_indices), batch_size=batch_size, shuffle=True)
            unlabeled_loader = DataLoader(Subset(new_train_set, unlabeled_indices), batch_size=batch_size, shuffle=True)

# The rest of the code remains unchanged (train_model, test_model functions, etc.)
# Define a function for training the model
def train_model(model, labeled_loader, optimizer, criterion):
    model.train()
    total_train_correct = 0
    total_train_samples = 0
    total_train_loss = 0

    for images, labels in labeled_loader:
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda() 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train_samples += labels.size(0)
        total_train_correct += (predicted == labels).sum().item()
        total_train_loss += loss.item()

    train_accuracy = 100 * total_train_correct / total_train_samples
    train_loss = total_train_loss / len(labeled_loader)
    return train_loss, train_accuracy


# Open a CSV file for writing results

    
    
for i in range(1, num_runs + 1):
  single_run(i)
    #     thread = threading.Thread(target=single_run, args=(i, writer))
    #     threads.append(thread)
    #     thread.start()

    # # Wait for all threads to finish
    # for thread in threads:
    #     thread.join()

print("All runs completed.")
