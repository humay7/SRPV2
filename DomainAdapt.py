import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import itertools
import os
from datetime import datetime

source_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomInvert(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

target_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

source_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=source_transform)
target_dataset = datasets.ImageFolder(root='./data/MNIST-M/test', transform=target_transform)

source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)

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
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    total_loss, total_correct = 0, 0
    
    for source_data, source_labels in source_loader:
        source_data, source_labels = source_data.to(device), source_labels.to(device)
        optimizer.zero_grad()
        source_features = model.feature_extractor(source_data)
        source_outputs = model.classifier(source_features)
        loss = criterion(source_outputs, source_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(source_outputs.data, 1)
        total_correct += (predicted == source_labels).sum().item()
    print(f'Epoch {epoch}, Train Loss: {total_loss/len(source_loader):.4f}, Train Acc: {100*total_correct/len(source_dataset):.2f}%')

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
save_path = f"models/PlainModel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")

def load_model(model_path):
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

#save_path = 'models/PlainModel_20250130_130831.pth'
model = load_model(save_path)
evaluate_loaded_model(model)