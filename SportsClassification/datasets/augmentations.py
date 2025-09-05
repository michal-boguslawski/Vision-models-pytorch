from torchvision.transforms import v2


AUGMENTATIONS_DICT = {
    "horizontal_flip": ("compose", v2.RandomHorizontalFlip(p=0.5)),
    "rotation": ("compose", v2.RandomRotation(degrees=(-45, 45))),
    "random_crop": ("compose", v2.RandomCrop(size=(224, 224))),
    "color_jitter": ("random", v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)),
    "random_resized_crop": ("compose", v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.1))),
    "grayscale": ("random", v2.RandomGrayscale(0.3)),
    "gaussian_blur": ("random", v2.GaussianBlur(5))
}
