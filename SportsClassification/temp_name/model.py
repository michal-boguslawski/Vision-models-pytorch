import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms

# implementation of alexnet in line with
# https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

class AlexNet(nn.Module):
    def __init__(self, n_classes, dims: int = 32, dropout: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.neural_network = nn.Sequential( # shape (3, 224, 224)
            transforms.Resize((224, 224)),
            nn.Conv2d(3, dims * 3, kernel_size=11, stride=4), # shape (96, 54, 54)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape (96, 26, 26)
            nn.Conv2d(dims * 3, dims * 8, kernel_size=5, padding=2), # shape (256, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape (256, 12, 12)
            nn.Conv2d(dims * 8, dims * 12, kernel_size=3, padding=1), # shape (384, 12, 12)
            nn.ReLU(),
            nn.Conv2d(dims * 12, dims * 12, kernel_size=3, padding=1), # shape (384, 12, 12)
            nn.ReLU(),
            nn.Conv2d(dims * 12, dims * 8, kernel_size=3, padding=1), # shape (256, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape (256, 5, 5)
            nn.Flatten(),
            nn.Linear(dims * 8 * 5 * 5, dims * 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dims * 128, dims * 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dims * 128, n_classes)
        )
        
    def forward(self, input_image):
        logits = self.neural_network(input_image)
        return logits
