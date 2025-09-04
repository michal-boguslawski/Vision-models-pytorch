import torch as T
import torch.nn as nn
from models import backbone, detection_head
from utils.helpers import filter_kwargs


# ------------------------------
# Predefined configs for VGG variants
# ------------------------------
PREDEFINED_CONFIGS = {
    "VGG11": {
        "num_layers_list": [1, 1, 2, 2, 2],
        "hidden_dims_list": [64, 128, 256, 512, 512],
    },
    "VGG13": {
        "num_layers_list": [2, 2, 2, 2, 2],
        "hidden_dims_list": [64, 128, 256, 512, 512],
    },
    "VGG16": {
        "num_layers_list": [2, 2, 3, 3, 3],
        "hidden_dims_list": [64, 128, 256, 512, 512],
    },
    "VGG19": {
        "num_layers_list": [2, 2, 4, 4, 4],
        "hidden_dims_list": [64, 128, 256, 512, 512],
    },
}


# ------------------------------
# Mapping of predefined configs to general architectures
# ------------------------------

CONFIG_TO_GENERAL = {
    "VGG16": "VGG",
    "VGG19": "VGG",
    "VGG8": "VGG",
}


# ------------------------------
# Backbone registry
# ------------------------------
BACKBONES_DICT = {
    "AlexNet": backbone.AlexNet,
    "VGG": backbone.VGG,  # generic VGG builder
}


# ------------------------------
# Detection head registry
# ------------------------------
DETECTION_HEADS_DICT = {
    "SimpleClassification": detection_head.SimpleClassificationHead,
    "AlexNet": detection_head.DoubleLinearHead,
    "VGG": detection_head.DoubleLinearHead,
    "GAP": detection_head.GAPDoubleLinearHead,
}


def _choose_backbone(
    backbone_name: str | None,
    backbone_kwargs: dict | None = None,
):
    backbone_kwargs = backbone_kwargs or {}
    
    if backbone_name is None:
        raise ValueError("backbone_name must be provided")
    
    general_name = CONFIG_TO_GENERAL.get(backbone_name, backbone_name)
    predefined_kwargs = PREDEFINED_CONFIGS.get(backbone_name, {})
    backbone_kwargs = {**predefined_kwargs, **backbone_kwargs}
    
    if general_name not in BACKBONES_DICT:
        raise KeyError(
            f"Unknown model_name: {general_name}. "
            f"Available backbones: {list(BACKBONES_DICT.keys())}"
        )
    model_cls = BACKBONES_DICT[general_name]
    backbone = model_cls(**filter_kwargs(model_cls, backbone_kwargs))
    return backbone


def _choose_detection_head(
    in_features: int,
    num_classes: int,
    detection_head_name: str | None,
    detection_head_kwargs: dict | None = None,
):
    detection_head_kwargs = detection_head_kwargs or {}
    
    if detection_head_name is None:
        raise ValueError("detection_head_name must be provided")
    general_name = CONFIG_TO_GENERAL.get(detection_head_name, detection_head_name)
    
    if general_name not in DETECTION_HEADS_DICT:
        raise KeyError(f"Unknown detection head: {general_name}. Available: {list(DETECTION_HEADS_DICT.keys())}")
    
    model_cls = DETECTION_HEADS_DICT[general_name]
    detection_head = model_cls(
        in_features=in_features,
        out_features=num_classes,
        **filter_kwargs(model_cls, detection_head_kwargs)
    )
    return detection_head


class BuildModel(nn.Module):
    """
    Build a model for classification/detection/segmentation.

    Args:
        n_classes: number of output classes
        model_name: optional, a pre-defined model name
        backbone_name: optional, backbone architecture
        detection_head_name: optional, head architecture
        backbone_kwargs: dict of extra args for backbone
        detection_head_kwargs: dict of extra args for detection head

    Returns:
        nn.Module: ready-to-train model
    """
    def __init__(
        self,
        num_classes: int,
        model_name: str | None = None,
        backbone_name: str | None = None,
        detection_head_name: str | None = None,
        backbone_kwargs: dict | None = None,
        detection_head_kwargs: dict | None = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.model_name = model_name
        self.backbone_name = backbone_name
        self.detecton_head_name = detection_head_name
        self.backbone_kwargs = backbone_kwargs
        self.detection_head_kwargs = detection_head_kwargs
        
        self.backbone = _choose_backbone(backbone_name=( model_name or backbone_name), backbone_kwargs=backbone_kwargs)
        self.detecton_head = _choose_detection_head(
            in_features=self.backbone.out_features,
            num_classes=num_classes,
            detection_head_name=( model_name or detection_head_name ),
            detection_head_kwargs=detection_head_kwargs
        )

    def forward(self, image: T.Tensor) -> T.Tensor:
        image_features = self.backbone(image)
        output = self.detecton_head(image_features)
        return output
