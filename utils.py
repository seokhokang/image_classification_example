import torch
from torch import nn

from torchvision import transforms as T


def get_augmentation(degree = 30, hshift = 0.3, vshift = 0.3, scale = 0.2, shear = 15, prob_hflip = 0, prob_vflip = 0):
    
    augmentation_operator = T.Compose([
        T.RandomAffine(degrees = degree, translate = (hshift, vshift), scale = (1-scale, 1+scale), shear = shear),
        T.RandomHorizontalFlip(prob_hflip),
        T.RandomVerticalFlip(prob_vflip),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return augmentation_operator
    
        
def get_no_augmentation():

    augmentation_operator = T.Compose([
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return augmentation_operator
