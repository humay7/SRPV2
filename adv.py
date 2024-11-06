import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import csv
import warnings
import random
import json
import torchvision.transforms.functional as F

warnings.filterwarnings("ignore")

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
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training settings
        self.k = 2  # Reduced for faster testing
        self.batch_size = 32
        self.gamma = 1.0
        self.learning_rate_max = 0.1  # Adjusted for quick training
        self.T_adv = 5  # Reduced for faster testing
        self.data_dir = './data/'

    def load_data(self):
        # Load a smaller subset of MNIST for faster training
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        full_data = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
        
        # Use a subset of the dataset (e.g., 10% of the training set)
        subset_size = int(0.1 * len(full_data))  # 10% of the dataset
        train_subset, _ = random_split(full_data, [subset_size, len(full_data) - subset_size])

        loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return loader

    def generate_adversarial_images(self):
        # Load the subset dataset
        source_train_loader = self.load_data()

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
                    adv_images = adv_images.detach().requires_grad_(True)

                # Detach the adversarial images from the computation graph before processing further
                adv_images = adv_images.detach()

                # Convert adversarial images to grayscale and resize to (28, 28)
                adv_images = torch.stack([transforms.Resize((28, 28))(F.rgb_to_grayscale(img)) for img in adv_images])
                adv_images = adv_images.squeeze(1)  # Remove the singleton channel dimension

                # Print shapes to confirm
                # print("Adjusted adversarial images shape:", adv_images.shape)

            counter_k += 1
            if counter_k >= self.k:
                break

        # Return the updated subset dataset loader
        return source_train_loader

# Instantiate and use the classes
model = Model()  # ResNet-18 modified for MNIST
train_ops = TrainOps(model)
train_set = train_ops.generate_adversarial_images()

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open('/content/SRPV2/parameters.json', 'r') as f:
    params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
images_per_class = params['images_per_class']
num_runs = params['num_runs']
n_samples_add_pool = params['n_samples_add_pool']

# Define active learning strategy (Uncertainty Sampling)
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

# Define transformations
channel_num = 3
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=channel_num),
    transforms.ToTensor(),                  
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

# Define the size of the smaller subset (e.g., 10% of the test set)
subset_size = int(0.1 * len(test_set))  # Adjust the percentage as needed

# Create the smaller test subset and assign it to test_set
test_set, _ = random_split(test_set, [subset_size, len(test_set) - subset_size])

# Print dataset stats
print('Dataset Stats:')
print('Train Dataset Size: {}, Test Dataset Size:{} \n'.format(len(train_set.dataset), len(test_set)))

# Define a function to perform a single run
def single_run(run_number, results_writer):
    # Initialize a new model for each run
    model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Modify the last fully connected layer for 10 classes
    model.cuda()  # Move model to GPU

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Access the original dataset from the DataLoader or Subset
    if isinstance(train_set, Subset):
        dataset = train_set.dataset  # Access the original dataset from the Subset
        if hasattr(dataset, 'dataset'):  # If it is nested (e.g., Subset of Subset)
            dataset = dataset.dataset
    else:
        dataset = train_set

    labeled_indices = []
    unlabeled_indices = []

    for class_label in range(10):
        # Access the targets from the original dataset directly
        class_indices = np.where(np.array(dataset.targets) == class_label)[0]
        np.random.shuffle(class_indices)
        selected_indices = class_indices[:min(images_per_class, len(class_indices))]
        labeled_indices.extend(selected_indices)

    labeled_set = Subset(dataset, labeled_indices)
    unlabeled_indices = list(set(range(len(dataset))) - set(labeled_indices))
    unlabeled_set = Subset(dataset, unlabeled_indices)

    # Create data loaders
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_model(model, labeled_loader, optimizer, criterion)
        test_accuracy = test_model(model, test_loader)
        print('Run {} -> Epoch [{}/{}], LP:{}, UP:{}, Train Acc: {:.2f}%, Loss: {:.6f}, Test Acc: {:.2f}%'.format(
            run_number, epoch, epochs, len(labeled_indices), len(unlabeled_indices), train_accuracy, train_loss, test_accuracy))
        
        # Store results
        results_writer.writerow([run_number, epoch, train_loss, train_accuracy, test_accuracy])

        # Avoid running uncertainty sampling during the last iteration
        if epoch < epochs:
            uncertain_indices = uncertainty_sampling(model, unlabeled_loader, n_samples_add_pool)
            labeled_indices.extend(uncertain_indices)
            unlabeled_indices = list(set(range(len(dataset))) - set(labeled_indices))
            labeled_loader = DataLoader(Subset(dataset, labeled_indices), batch_size=batch_size, shuffle=True)
            unlabeled_loader = DataLoader(Subset(dataset, unlabeled_indices), batch_size=batch_size, shuffle=True)

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

# Define a function for testing the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda() 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    return test_accuracy

# Open a CSV file for writing results
with open('resnet_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Run', 'Epoch', 'Train Loss', 'Train Accuracy', 'Test Accuracy'])

    # Start parallel runs
    for i in range(1, num_runs + 1):
        single_run(i, writer)

print("All runs completed.")
