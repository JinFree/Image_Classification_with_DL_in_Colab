import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class MODEL(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.mobilenet_v2(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout()
            , nn.Linear(1000, num_classes)
            , nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x = self.network(x)
        return self.classifier(x)