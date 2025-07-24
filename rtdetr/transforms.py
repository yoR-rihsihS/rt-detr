import torch
from torchvision.transforms import v2

def get_train_transforms(size=640):
    return v2.Compose([
        v2.ToImage(),
        
        v2.RandomChoice([
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=10, fill=0, interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomPerspective(distortion_scale=0.5, p=0.3, fill=0, interpolation=v2.InterpolationMode.BILINEAR),
        ], [0.7, 0.3]),

        v2.RandomChoice([
            v2.Identity(),
            v2.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0), ratio=(0.75, 1.25), interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomCrop(size=(size, size), pad_if_needed=True, padding_mode='constant', fill=0),
        ], [0.5, 0.4, 0.1]),
        
        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomApply([v2.ColorJitter(brightness=0.5, contrast=0.5)], p=0.5),
        v2.RandomApply([v2.ColorJitter(saturation=0.5, hue=0.2)], p=0.2),

        v2.RandomApply([v2.GaussianBlur(kernel_size=13, sigma=(0.1, 5.0))], p=0.2),

        v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR),

        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_valid_transforms(size=640):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
