import pytest
import torch as T
from models.backbone import AlexNet, ResNet


class TestModelsBackbone:
    def test_alexnet_output_shape(self):
        alexnet_backbone = AlexNet()
        
        sample_image = T.rand((1, 3, 224, 224))
        with T.no_grad():
            output = alexnet_backbone(sample_image)
        
        assert output.shape == (1, 256, 5, 5)
 
    def test_resnet_output_shape(self):
        resnet_backbone = ResNet()
        
        sample_image = T.rand((1, 3, 224, 224))
        with T.no_grad():
            output = resnet_backbone(sample_image)
        
        assert output.shape == (1, 512, 7, 7)
