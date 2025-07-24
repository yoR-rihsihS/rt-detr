import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights

class BackBone(nn.Module):
    """
    Backbone network for feature extraction.
    Args:
        - model (str): Backbone model name. Options are 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
        - num_levels (int): Number of levels (starting from last level) to extract features from. Default is 3. Must be less than 5.
    Returns:
        - outputs (list): List of feature maps from the backbone network.
    """
    def __init__(self, model='resnet50', num_levels=3):
        super(BackBone, self).__init__()
        if model == 'resnet18':
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 1.81 GFlops
        elif model == 'resnet34':
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)  # 3.66 GFlops
        elif model == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # 4.09 GFlops
        elif model == 'resnet101':
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)  # 7.8 GFlops
        elif model == 'resnet152':
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)  # 11.51 GFlops
        else:
            raise ValueError(f"{model} is not supported. Choose from 'resnet18', 'resnet34' 'resnet50', 'resnet101', or 'resnet152'")
        assert num_levels < 5, f"num_levels must be less than 5, but got {num_levels}"
        self.num_levels = num_levels
        self.layers = nn.ModuleList([
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
            ),                          # 1/4, 64
            model.layer1,               # 1/4, 256 or 64
            model.layer2,               # 1/8, 512 or 128
            model.layer3,               # 1/16, 1024 or 256
            model.layer4,               # 1/32, 2048 or 512
        ])
        del model

    def forward(self, image):
        outputs = [image]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        return outputs[-self.num_levels:]