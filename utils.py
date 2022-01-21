import torch
from torch import nn

from torchvision import transforms as T


def get_data_augmentation(degree = 30, hshift = 0.3, vshift = 0.3, scale = 0.2, brightness = 0.2, contrast = 0.2, prob_hflip = 0.5, prob_vflip = 0):
    
    augmentation_operators = nn.Sequential(
        T.RandomAffine(degrees = degree, translate = (hshift, vshift), scale = (1-scale, 1+scale)),
        T.ColorJitter(brightness = brightness, contrast = contrast),
        T.RandomHorizontalFlip(prob_hflip),
        T.RandomVerticalFlip(prob_vflip),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )
    data_augmentation = torch.jit.script(augmentation_operators)
    
    return data_augmentation
    
    
def get_no_augmentation():

    no_augmentation = torch.jit.script(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    return no_augmentation