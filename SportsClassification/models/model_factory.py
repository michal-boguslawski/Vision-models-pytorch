"""
🔹 Purpose of model_factory.py

Create any model given a name or configuration.

Load pretrained weights if needed.

Apply custom heads (e.g., classification, detection, segmentation).

Return ready-to-train models for training or inference.
"""
import torch.nn as nn
from models import backbone, detection_head


BACKBONES_DICT = {
    "AlexNet": backbone.AlexNet
}


DETECTION_HEADS_DICT = {
    "SimpleClassification": detection_head.SimpleClassificationHead,
    "AlexNet": detection_head.SimpleClassificationHead,
}


def build_model(
    num_classes: int,
    model_name: str | None = None,
    backbone_name: str | None = None,
    detection_head_name: str | None = None,
    backbone_kwargs: dict | None = None,
    detection_head_kwargs: dict | None = None,
    *args,
    **kwargs
 )-> nn.Module:
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
    backbone_kwargs = backbone_kwargs or {}
    detection_head_kwargs = detection_head_kwargs or {}

    if model_name:
        if model_name not in BACKBONES_DICT:
            raise KeyError(
                f"Unknown model_name: {model_name}. "
                f"Available backbones: {list(BACKBONES_DICT.keys())}, "
                f"Available heads: {list(DETECTION_HEADS_DICT.keys())}"
            )
        backbone = BACKBONES_DICT[model_name](**backbone_kwargs)
        detection_head = DETECTION_HEADS_DICT[model_name](
            in_features=backbone.out_features,
            out_features=num_classes,
            **detection_head_kwargs
        )
        return nn.Sequential(backbone, detection_head)
    elif backbone_name and detection_head_name:
        if backbone_name not in BACKBONES_DICT:
            raise KeyError(f"Unknown backbone: {backbone_name}. Available: {list(BACKBONES_DICT.keys())}")
        if detection_head_name not in DETECTION_HEADS_DICT:
            raise KeyError(f"Unknown detection head: {detection_head_name}. Available: {list(DETECTION_HEADS_DICT.keys())}")
        
        backbone = BACKBONES_DICT[backbone_name](**backbone_kwargs)
        detection_head = DETECTION_HEADS_DICT[detection_head_name](
            in_features=backbone.out_features,
            out_features=num_classes,
            **detection_head_kwargs
        )
        return nn.Sequential(backbone, detection_head)
    else:
        raise ValueError("You must provide either `model_name` or both `backbone_name` and `detection_head_name`.")
