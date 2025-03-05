import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import itertools
import os
from datetime import datetime

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



source_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

target_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


source_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=source_transform)
target_dataset = datasets.ImageFolder(root='./data/MNIST-M/testing', transform=target_transform)




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

class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DomainAdaptationModel().to(device)
criterion = nn.CrossEntropyLoss()
coral_loss = CORALLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
T_adv = 5 

adv_dataset = generate_adversarial_images(model, labeled_loader, criterion, gamma, T_adv)
combined_source_dataset = ConcatDataset([source_dataset, adv_dataset])


source_loader = DataLoader(combined_source_dataset, batch_size=64, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)

def train(epoch):
    model.train()
    total_loss, total_correct = 0, 0
    
    if len(source_loader) > len(target_loader):
        target_iter = itertools.cycle(target_loader)
        data_iterator = zip(source_loader, target_iter)
    else:
        source_iter = itertools.cycle(source_loader)
        data_iterator = zip(source_iter, target_loader)
    
    for (source_data, source_labels), (target_data, _) in data_iterator:
        source_data, source_labels = source_data.to(device), source_labels.to(device)
        target_data = target_data.to(device)
        optimizer.zero_grad()
        source_features = model.feature_extractor(source_data)
        target_features = model.feature_extractor(target_data)
        source_outputs = model.classifier(source_features)
        loss = criterion(source_outputs, source_labels) + 0.00 * coral_loss(source_features, target_features)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(source_outputs.data, 1)
        total_correct += (predicted == source_labels).sum().item()
    print(f'Epoch {epoch}, Train Loss: {total_loss/len(source_loader):.4f}, Train Acc: {100*total_correct/len(combined_source_dataset):.2f}%')

def test():
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for data, labels in target_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
    print(f'Test Loss: {total_loss/len(target_loader):.4f}, Test Acc: {100*total_correct/len(target_dataset):.2f}%')

for epoch in range(1, 11):
    train(epoch)
    test()

os.makedirs("models", exist_ok=True)
save_path = f"models/CORAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")


# Load the model and evaluate
"""def load_model(model_path):
    model = DomainAdaptationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_loaded_model(model):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for data, labels in target_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
    print(f'Loaded Model Test Loss: {total_loss/len(target_loader):.4f}, Test Acc: {100*total_correct/len(target_dataset):.2f}%')

model = load_model(save_path)
evaluate_loaded_model(model)"""
