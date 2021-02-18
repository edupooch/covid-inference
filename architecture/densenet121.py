import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

def load_densenet121(): 
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 2),
        nn.Softmax(dim=1))
    return model
