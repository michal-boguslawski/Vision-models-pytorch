import torch as T
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Implementation of AlexNet according to paper
    """
    def __init__(
        self,
        in_channels: int = 3,
        activation_fn: nn.Module = nn.ReLU,
        pool_type: nn.Module = nn.MaxPool2d,
        dropout: float = 0.5,
        out_features: int = 1000,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Input size expected (3x224x224)"""
        self.in_channels = in_channels
        self.activation_fn = activation_fn
        self.pool_type = pool_type
        self.dropout = dropout
        self.out_features = out_features
        
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=0,
            ),
            activation_fn(inplace=True),
            pool_type(kernel_size=3, stride=2),
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                padding=2,
            ),
            activation_fn(inplace=True),
            pool_type(kernel_size=3, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                padding=1,
            ),
            activation_fn(inplace=True),
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                padding=1,
            ),
            activation_fn(inplace=True),
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            activation_fn(inplace=True),
            pool_type(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(5 * 5 * 256, 4096),
            activation_fn(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            activation_fn(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, out_features),
        )
    
    def forward(self, image: T.Tensor) -> T.Tensor:
        assert image.shape[-3:] == (3, 224, 224)
        x = self.features(image)
        x = T.flatten(x, 1)
        x = self.classifier(x)
        return x
