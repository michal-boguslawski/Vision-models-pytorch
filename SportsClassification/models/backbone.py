from collections import OrderedDict
from typing import Type, List
import torch as T
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Implementation of AlexNet according to paper
    """
    def __init__(
        self,
        in_channels: int = 3,
        activation_fn: Type[nn.Module] = nn.ReLU,
        pool_type: Type[nn.Module] = nn.MaxPool2d,
        dropout: float = 0.5,
        if_flatten: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Input size expected (3x224x224)"""
        self.in_channels = in_channels
        self.activation_fn = activation_fn
        self.pool_type = pool_type
        self.dropout = dropout
        self.out_features = (256, 5, 5)

        self.local_resp_norm_kwargs = {
            "size": 5,
            "k": 2,
            "alpha": 1e-4,
            "beta": 0.75
        }
        
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=0),
            activation_fn(inplace=True),
            nn.LocalResponseNorm(**self.local_resp_norm_kwargs),
            pool_type(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            activation_fn(inplace=True),
            nn.LocalResponseNorm(**self.local_resp_norm_kwargs),
            pool_type(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            activation_fn(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            activation_fn(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            activation_fn(inplace=True),
            pool_type(kernel_size=3, stride=2),
        ]

        self.features = nn.Sequential(*layers)
    
    def forward(self, image: T.Tensor) -> T.Tensor:
        assert image.shape[-3:] == (3, 224, 224)
        x = self.features(image)
        return x


class VGG(nn.Module):
    """
    Implementation of VGG16 according to paper
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_layers_list: List[int] = [1, 1, 2, 2, 2],
        hidden_dims_list: List[int] = [64, 128, 256, 512, 512],
        activation_fn: Type[nn.Module] = nn.ReLU,
        pool_type: Type[nn.Module] = nn.MaxPool2d,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Input size expected (3x224x224)"""
        self.in_channels = in_channels
        self.activation_fn = activation_fn
        self.pool_type = pool_type
        self.num_layers_list = num_layers_list
        self.hidden_dims_list = hidden_dims_list
        self.out_features = (512, 7, 7)
        
        list_of_layers = self._prebuild_list_of_layers(
            in_channels=in_channels,
            num_layers_list=num_layers_list,
            hidden_dims_list=hidden_dims_list,
            activation_fn=activation_fn,
            pool_type=pool_type
        )
        
        self.features = nn.Sequential(OrderedDict(list_of_layers))

    @staticmethod
    def _prebuild_list_of_layers(
        in_channels: int,
        num_layers_list: List[int],
        hidden_dims_list: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        pool_type: Type[nn.Module] = nn.MaxPool2d,
    ):
        list_of_layers = []
        prev_features = in_channels
        
        for i, (num_layers, hidden_dims) in enumerate(zip(num_layers_list, hidden_dims_list)):
            # add specified number of layers to the block
            for n in range(num_layers):
                list_of_layers.append(
                    (f"block{i}conv{n}", nn.Conv2d(prev_features, hidden_dims, 3, 1, 1))
                )
                list_of_layers.append(
                    (f"block{i}activation{n}", activation_fn(inplace=True))
                )
                prev_features = hidden_dims

            # at the end of each block apply max pool
            list_of_layers.append(
                (f"block{i}pool", pool_type(kernel_size=2, stride=2))
            )
        return list_of_layers
    
    def forward(self, image: T.Tensor) -> T.Tensor:
        assert image.shape[-3:] == (3, 224, 224)
        x = self.features(image)
        return x
