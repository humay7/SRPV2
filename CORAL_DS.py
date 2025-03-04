import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
from torchvision import datasets, transforms, models
import itertools
import os
from datetime import datetime

# ------------------------
# Define the AdversarialDataAugment module
# ------------------------
class AdversarialDataAugment(nn.Module):
    def __init__(self, model, criterion, gamma=1.0, T_adv=1):
        """
        Initialize the adversarial data augmentation module.
        
        Args:
            model (nn.Module): Pretrained model used to compute gradients.
            criterion: Loss function (e.g., nn.CrossEntropyLoss()).
            gamma (float): Step size for the adversarial perturbation.
            T_adv (int): Number of FGSM iterations.
        """
        super(AdversarialDataAugment, self).__init__()
        self.model = model
        self.criterion = criterion
        self.gamma = gamma
        self.T_adv = T_adv

    def forward(self, x, labels):
        """
        Apply FGSM adversarial perturbation to the input images.
        
        Args:
            x (Tensor): Input image batch (assumed to be in [0, 1]).
            labels (Tensor): True labels corresponding to x.
        
        Returns:
            Tensor: Adversarially perturbed images.
        """
        x_adv = x.clone().detach().requires_grad_(True)
        for _ in range(self.T_adv):
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.zero_()
            loss.backward()
            x_adv = x_adv + self.gamma * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv.detach(), 0.0, 1.0).requires_grad_(True)
        return x_adv.detach()

# ------------------------
# Define CORAL Loss
# ------------------------
class CORALLoss(nn.Module):
    def __init__(self):
        super(CORALLoss, self).__init__()
    def forward(self, source, target):
        d = source.size(1)
        source = source.view(source.size(0), -1)
        target = target.view(target.size(0), -1)
        source_cov = torch.mm(source.t(), source) / (source.size(0) - 1)
        target_cov = torch.mm(target.t(), target) / (target.size(0) - 1)
        return torch.norm(source_cov - target_cov, p='fro')**2 / (4 * d * d)

# ------------------------
# Define the domain adaptation model
# ------------------------
class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# ------------------------
# Set up device, model, loss functions, optimizer, and adversarial augmenter
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DomainAdaptationModel().to(device)
criterion = nn.CrossEntropyLoss()
coral_loss = CORALLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create an instance of the adversarial augmenter.
# (We use FGSM with gamma=0.1 and T_adv=5 iterations)
adv_augment = AdversarialDataAugment(model, criterion, gamma=0.1, T_adv=5)

# ------------------------
# Define transformation pipelines (without any inline adversarial transform)
# ------------------------
# For source dataset, we add a target_transform that converts labels to tensors
source_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# Convert labels (which are normally ints) to tensors.
source_target_transform = lambda x: torch.tensor(x)

target_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ------------------------
# Load datasets
# ------------------------
source_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                transform=source_transform, target_transform=source_target_transform)
target_dataset = datasets.ImageFolder(root='./data/MNIST-M/testing', transform=target_transform)

source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)

# ------------------------
# Generate an adversarial dataset from the source dataset using the augmenter
# ------------------------
def generate_adversarial_dataset(loader, augmenter):
    adv_images_list = []
    adv_labels_list = []
    model.eval()  # Ensure the model is in evaluation mode
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = augmenter(images, labels)
        adv_images_list.append(adv_images.cpu())
        adv_labels_list.append(labels.cpu())
    adv_images_tensor = torch.cat(adv_images_list, dim=0)
    adv_labels_tensor = torch.cat(adv_labels_list, dim=0)
    return TensorDataset(adv_images_tensor, adv_labels_tensor)

adv_dataset = generate_adversarial_dataset(source_loader, adv_augment)

# Optionally, combine the original source dataset with the adversarial dataset
combined_dataset = ConcatDataset([source_dataset, adv_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

# ------------------------
# Define training and testing loops (training on the combined dataset)
# ------------------------
def train(epoch):
    model.train()
    total_loss, total_correct = 0, 0
    for images, labels in combined_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features = model.feature_extractor(images)
        outputs = model.classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch}, Train Loss: {total_loss/len(combined_loader):.4f}, "
          f"Train Acc: {100*total_correct/len(combined_dataset):.2f}%")

def test():
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for images, labels in target_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
    print(f"Test Loss: {total_loss/len(target_loader):.4f}, "
          f"Test Acc: {100*total_correct/len(target_dataset):.2f}%")

# ------------------------
# Run training and testing
# ------------------------
for epoch in range(1, 11):
    train(epoch)
    test()

os.makedirs("models", exist_ok=True)
save_path = f"models/CORAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")
