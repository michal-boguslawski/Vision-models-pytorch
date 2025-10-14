from torchvision.transforms import v2
from torchvision.transforms import RandomApply

AUGMENTATIONS_DICT = {
    # Geometric / spatial transforms
    "horizontal_flip": v2.RandomHorizontalFlip(p=0.5),
    "rotation": v2.RandomRotation(degrees=(-15, 15)),  # reduced rotation
    "random_crop": v2.RandomCrop(size=(224, 224)),
    "random_resized_crop": v2.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.1)),
    "affine": RandomApply([v2.RandomAffine(
        degrees=(-15, 15),
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        shear=(-5, 5)
    )], p=0.5),
    "perspective": RandomApply([v2.RandomPerspective(distortion_scale=0.2, p=1)], p=0.5),

    # Color / brightness / contrast transforms
    "color_jitter": RandomApply([v2.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    )], p=0.5),
    "grayscale": v2.RandomGrayscale(p=0.5),

    # Other image transforms
    "gaussian_blur": RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.5),
    "erasing": v2.RandomErasing(
        p=0.5,
        scale=(0.02, 0.15),
        ratio=(0.3, 3.3),
        value=0
    ),

    # Extreme color transforms (less frequent)
    "invert": v2.RandomInvert(p=0.2),
    "posterize": v2.RandomPosterize(bits=2, p=0.2),
    "solarize": v2.RandomSolarize(threshold=192.0, p=0.2),
    "adjust_sharpness": v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    "autocontrast": v2.RandomAutocontrast(p=0.3),
    "equalize": v2.RandomEqualize(p=0.3),
}
