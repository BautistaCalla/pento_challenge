import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

class CachedImageFolder(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        
        sample, target = super().__getitem__(index)
        self.cache[index] = (sample, target)
        return sample, target
    
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)  # No transform here
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        path, target = self.samples[index]
        if index not in self.cache:
            sample = self.loader(path)
            self.cache[index] = sample
        else:
            sample = self.cache[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

def get_dataset(data_dir, augmented=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_augmented = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    dataset = CustomImageFolder(root=data_dir, transform=transform_augmented) if augmented else CachedImageFolder(root=data_dir, transform=transform)
    return dataset