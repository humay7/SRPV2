import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, TensorDataset
import numpy as np
import csv
import warnings
import random
import json

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

# Load and preprocess data
def load_data():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    target_transform = transforms.Lambda(lambda y: torch.tensor(y))

    # Load the MNIST dataset and apply the transforms
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform  # Ensure labels are tensors
    )
        
    return dataset

# Generate adversarial images
# Generate adversarial images
def generate_adversarial_images(model, labeled_loader, criterion, gamma, T_adv):
    """
    Generate adversarial images from 50% of the labeled set.
    """
    model.eval()  # Set the model to evaluation mode for adversarial generation

    # Determine half of the labeled dataset
    half_size = len(labeled_loader.dataset) // 2
    subset_indices = list(range(half_size))  # Take the first half
    half_loader = DataLoader(
        Subset(labeled_loader.dataset, subset_indices), 
        batch_size=labeled_loader.batch_size, 
        shuffle=False
    )

    adv_images_list = []
    adv_labels_list = []

    for images, labels in half_loader:
        images, labels = images.cuda(), labels.cuda()

        # Initialize adversarial images
        adv_images = images.clone().detach().requires_grad_(True)

        # Perform T_adv gradient ascent steps
        for _ in range(T_adv):
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            adv_images = adv_images + gamma * adv_images.grad.sign()

            # Clip and re-attach gradient
            adv_images = torch.clamp(adv_images.detach(), min=0.0, max=1.0).requires_grad_(True)

        adv_images_list.append(adv_images.cpu())
        adv_labels_list.append(labels.cpu())

    # Combine generated adversarial images and labels
    adv_images_tensor = torch.cat(adv_images_list, dim=0)
    adv_labels_tensor = torch.cat(adv_labels_list, dim=0)

    return TensorDataset(adv_images_tensor, adv_labels_tensor)


# Main training loop with active learning and adversarial image generation
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

k = 1  # Reduced for faster testing
batch_size = 32
gamma = 1.0
learning_rate_max = 0.1  # Adjusted for quick training
T_adv = 5  # Reduced for faster testing
T_min = 50






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
train_set = load_data()
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)


# Print dataset stats
print('Dataset Stats:')
print('Train Dataset Size: {}, Test Dataset Size: {}'.format(len(train_set), len(test_set)))

# Define a function to perform a single run
def single_run(train_set, run_number, results_writer):
    model = Model().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Directly use train_set without accessing train_set.dataset
    # print('Train Set Length:', len(train_set))
    
    labeled_indices = []
    unlabeled_indices = []
    for class_label in range(10):
        class_indices = np.where(np.array(train_set.targets) == class_label)[0]
        np.random.shuffle(class_indices)
        selected_indices = class_indices[:min(images_per_class, len(class_indices))]
        labeled_indices.extend(selected_indices)
    labeled_set = Subset(train_set, labeled_indices)
    unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices))
    unlabeled_set = Subset(train_set, unlabeled_indices)

    # Create data loaders
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_model(model, labeled_loader, optimizer, criterion)
        test_accuracy = test_model(model, test_loader)
        print(f'Run {run_number} -> Epoch [{epoch}/{epochs}], LP: {len(labeled_indices)}, UP: {len(unlabeled_indices)}, '
              f'Train Acc: {train_accuracy:.2f}%, Loss: {train_loss:.6f}, Test Acc: {test_accuracy:.2f}%')
        
        # Store results
        results_writer.writerow([run_number, epoch, train_loss, train_accuracy, test_accuracy])
        
        if epoch < epochs:  # Avoid Running Uncertainty Sampling during the last iteration
            # Generate adversarial dataset and update unlabeled set
            adv_dataset = generate_adversarial_images(model, labeled_loader, criterion, gamma, T_adv)
            train_set = ConcatDataset([train_set, adv_dataset])  # Update train_set
            unlabeled_set = ConcatDataset([unlabeled_set, adv_dataset])  # Update unlabeled set
            
            # **Create a new DataLoader for unlabeled_set after updating it**
            unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
            
            # Perform uncertainty sampling
            uncertain_indices = uncertainty_sampling(model, unlabeled_loader, n_samples_add_pool)
            labeled_indices.extend(uncertain_indices)
            
            # Recalculate unlabeled and labeled indices
            labeled_indices = list(set(labeled_indices))  # Ensure no duplicates in labeled_indices
            unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices))
            
            # Update DataLoaders for labeled and unlabeled sets
            labeled_loader = DataLoader(Subset(train_set, labeled_indices), batch_size=batch_size, shuffle=True)
            unlabeled_loader = DataLoader(Subset(train_set, unlabeled_indices), batch_size=batch_size, shuffle=True)

# Define a function for training the model
def train_model(model, labeled_loader, optimizer, criterion):
    model.train()
    total_train_correct = 0
    total_train_samples = 0
    total_train_loss = 0

    for images, labels in labeled_loader:
      
        print(f"Batch size: {images.size(0)}")  # Check actual batch size

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


def check_tensors_in_subset(train_subset):
    for i, (image, label) in enumerate(train_subset):
        if not isinstance(image, torch.Tensor):
            print(f"Image at index {i} is not a tensor: {type(image)}")
        if not isinstance(label, torch.Tensor):
            print(f"Label at index {i} is not a tensor: {type(label)}")
        if i >= 5:  # Check the first few samples and then break
            break
    print("Check complete.")





# Open a CSV file for writing results
with open('resnet_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Run', 'Epoch', 'Train Loss', 'Train Accuracy', 'Test Accuracy'])

    # Start parallel runs
    for i in range(1, num_runs + 1):
        single_run(train_set, i, writer)

print("All runs completed.")

