from torchvision.transforms import v2
from typing import Any


AUGMENTATIONS_DICT: dict[str, tuple[str, Any]] = {
    "horizontal_flip": ("compose", v2.RandomHorizontalFlip(p=0.5)),
    "rotation": ("compose", v2.RandomRotation(degrees=(-45, 45))),
    "random_crop": ("compose", v2.RandomCrop(size=(224, 224))),
    "color_jitter": ("random", v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)),
    "random_resized_crop": ("compose", v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.1))),
    "grayscale": ("random", v2.RandomGrayscale(0.3)),
    "gaussian_blur": ("random", v2.GaussianBlur(5)),
    "perspective": ("random", v2.RandomPerspective()),
    "affine": ("random", v2.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1), scale=(0.8, 1.1), shear=(-10, 10))),
    "erasing": ("random", v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)),
    "invert": ("random", v2.RandomInvert()),
    "posterize": ("random", v2.RandomPosterize(bits=2)),
    "solarize": ("random", v2.RandomSolarize(threshold=192.0)),
    "adjust_sharpness": ("random", v2.RandomAdjustSharpness(sharpness_factor=2)),
    "autocontrast": ("random", v2.RandomAutocontrast()),
    "equalize": ("random", v2.RandomEqualize()),
}
