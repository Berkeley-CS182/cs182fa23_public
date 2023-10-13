import torch
import torch.nn as nn
import torchvision

class BasicConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class ResNet18(nn.Module):
    # Keep in mind that you will need to resize the Image to 224x224
    def __init__(self):
        super().__init()
        self.backbone = torchvision.models.resnet18()
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_ftrs, 10)
    def forward(self, x):
        return self.backbone(x)


class MLP(nn.Module):
    def __init__(self, num_layers=7, size=2048, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(3072, size)
        self.hidden = self.hidden = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.out = nn.Linear(size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)
        x = self.out(x)
        return x
        
        
