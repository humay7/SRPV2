import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import configparser

# Define the model using ResNet-18
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)  # Modify for 10 classes

    def forward(self, x):
        return self.model(x)

class TrainOps:
    def __init__(self, model, exp_dir):
        self.model = model
        self.exp_dir = exp_dir

        # Device configuration (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load the experiment configuration
        self.load_exp_config()

        self.data_dir = './data/'
        
    def load_exp_config(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.exp_dir, 'exp_configuration'))

        # Experiment settings
        self.source_dataset = config.get('EXPERIMENT_SETTINGS', 'source_dataset')
        self.no_images = config.getint('EXPERIMENT_SETTINGS', 'no_images')

        # Training settings
        self.k = config.getint('MAIN_SETTINGS', 'k')
        self.batch_size = config.getint('MAIN_SETTINGS', 'batch_size')
        self.gamma = config.getfloat('MAIN_SETTINGS', 'gamma')
        self.learning_rate_max = config.getfloat('MAIN_SETTINGS', 'learning_rate_max')
        self.T_adv = config.getint('MAIN_SETTINGS', 'T_adv')
        self.T_min = config.getint('MAIN_SETTINGS', 'T_min')

    def load_data(self, dataset, split='train'):
        # Load data with transformations
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if dataset == 'mnist':
            data = torchvision.datasets.MNIST(self.data_dir, train=(split == 'train'), download=True, transform=transform)
        elif dataset == 'svhn':
            data = torchvision.datasets.SVHN(self.data_dir, split=split, download=True, transform=transform)

        loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        return loader

    def generate_adversarial_images(self):
        # Load dataset
        source_train_loader = self.load_data(self.source_dataset, split='train')

        # Loss function and optimizer for adversarial training
        criterion = nn.CrossEntropyLoss()
        max_optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate_max)

        counter_k = 0

        for t in range(self.k):  # Loop over adversarial training steps
            for images, labels in source_train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Initialize adversarial images
                adv_images = images.clone().detach().requires_grad_(True)

                for n in range(self.T_adv):  # Gradient ascent for adversarial training
                    max_optimizer.zero_grad()

                    # Forward pass for adversarial images
                    adv_outputs = self.model(adv_images)
                    
                    # max_loss_1: Classification loss on adversarial images
                    max_loss_1 = criterion(adv_outputs, labels)
                    
                    # Backward and update adversarial images
                    max_loss_1.backward()
                    adv_images = adv_images + self.gamma * adv_images.grad.sign()

                # Update the original dataset with adversarial images
                source_train_loader.dataset.data = torch.cat((source_train_loader.dataset.data, adv_images.cpu()))
                source_train_loader.dataset.targets = torch.cat((source_train_loader.dataset.targets, labels.cpu()))

            counter_k += 1
            if counter_k >= self.k:
                break

        # Return the updated dataset
        return source_train_loader


model = Model()  # ResNet-18 modified for MNIST
exp_dir = './experiment'  # Directory containing `exp_configuration`

train_ops = TrainOps(model, exp_dir)
updated_dataset_loader = train_ops.generate_adversarial_images()
