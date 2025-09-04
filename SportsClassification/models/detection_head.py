"""
🔹 Purpose of detection_head.py

Take the feature map from the backbone (e.g., CNN or transformer features).

Predict bounding boxes, class scores, and optionally objectness.

Usually contains:

Fully connected layers or convolutional layers.

Anchor handling (if using anchor-based methods like Faster R-CNN).

Activation functions (sigmoid/softmax) for probabilities.

Optional post-processing (e.g., NMS) in inference.
"""
import math
from typing import Type
import torch as T
import torch.nn as nn


class DoubleLinearHead(nn.Module):
    def __init__(
        self,
        in_features: int | tuple,
        out_features: int = 1000,
        activation_fn: Type[nn.Module] = nn.ReLU,
        hidden_dims: int = 4096,
        dropout: float = 0.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.in_features = math.prod(in_features)
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_features, hidden_dims),
            activation_fn(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            activation_fn(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, out_features)
        )

    def forward(self, input_tensor: T.Tensor) -> T.Tensor:
        return self.classifier(input_tensor)


class GAPDoubleLinearHead(nn.Module):
    def __init__(
        self,
        in_features: tuple,
        out_features: int = 1000,
        activation_fn: Type[nn.Module] = nn.ReLU,
        hidden_dims: int = 4096,
        dropout: float = 0.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.in_features = in_features[0]
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        self.classifier = nn.Sequential(
            nn.AvgPool2d(in_features[1:]),
            nn.Flatten(1),
            nn.Linear(self.in_features, hidden_dims),
            activation_fn(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            activation_fn(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, out_features)
        )

    def forward(self, input_tensor: T.Tensor) -> T.Tensor:
        return self.classifier(input_tensor)


class SimpleClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int | tuple,
        out_features: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.in_features = math.prod(in_features)
        self.out_features = out_features
        
        self.head = nn.Linear(
            in_features=in_features,
            out_features=out_features
        )

    def forward(self, input_tensor: T.Tensor) -> T.Tensor:
        return self.head(input_tensor)
