import math
import torch 
import json
import torch.nn as nn

def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))

def bias_init_with_prob(prior_prob=0.01):
    """
    Initialize conv/fc bias value according to a given probability value.
    """
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init

def get_activation(act, inpace=True):
    act = act.lower()
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'gelu':
        m = nn.GELU() 
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError(f"Unknown activation {act} requested")  
    if hasattr(m, 'inplace'):
        m.inplace = inpace
    return m

def collate_fn(batch):
    targets = []
    images = []
    for image, target in batch:
        targets.append(target)
        images.append(image)
    return torch.stack(images), targets

def get_label_to_name(coco_annotations_file_path):
    file_dict = json.load(open(coco_annotations_file_path, 'r'))
    label_to_name = {i: cat_dict['name'] for i, cat_dict in enumerate(file_dict['categories'])}
    label_to_name[len(file_dict['categories'])] = 'background'  # Add 81 as background class
    return label_to_name

def get_label_to_coco_id(coco_annotations_file_path):
    file_dict = json.load(open(coco_annotations_file_path, 'r'))
    label_to_coco_id = {cat_dict['id']: i for i, cat_dict in enumerate(file_dict['categories'])}
    return label_to_coco_id