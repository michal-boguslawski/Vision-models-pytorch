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

import torch as T
import torch.nn as nn


class SimpleClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        
        self.head = nn.Linear(
            in_features=in_features,
            out_features=out_features
        )

    def forward(self, input_tensor: T.Tensor) -> T.Tensor:
        assert input_tensor.dim() <= 2
        return self.head(input_tensor)
