import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.model = models.resnet50(pretrained=True)

        # Freeze earlier layers to prevent overfitting
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the final layer for custom classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Model Initialization

def initialize_model(num_classes=38, learning_rate=0.001):  # Set num_classes=38
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlantDiseaseModel(num_classes).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.fc.parameters(), lr=learning_rate)

    return model, criterion, optimizer



