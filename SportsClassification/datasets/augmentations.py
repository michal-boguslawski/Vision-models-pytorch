from torchvision.transforms import v2


AUGMENTATIONS_DICT = {
    "horizontal_flip": v2.RandomHorizontalFlip(p=0.5),
    "rotation": v2.RandomRotation(degrees=(0, 90)),
    "random_crop": v2.RandomCrop(size=(224, 224)),
    "color_jitter": v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
}
