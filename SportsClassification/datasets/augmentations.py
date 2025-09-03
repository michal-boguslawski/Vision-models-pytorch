from torchvision.transforms import v2


AUGMENTATIONS_DICT = {
    "horizontal_flip": v2.RandomHorizontalFlip(p=0.5),
    "rotation": v2.RandomRotation(degrees=(-45, 45)),
    "random_crop": v2.RandomCrop(size=(224, 224)),
    "color_jitter": v2.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.1),
    "random_resized_crop": v2.RandomResizedCrop(size=(224, 224), scale=(0.5, 1))
}
