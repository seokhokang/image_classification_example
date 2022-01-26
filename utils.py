import torch
from torch import nn

from torchvision import transforms as T


def get_augmentation(prob_equal = 0, prob_invert = 0, brightness = 1, contrast = 1, saturation = 1, degree = 30, hshift = 0.3, vshift = 0.3, scale = 0.2, shearx = 0, sheary = 0, prob_hflip = 0, prob_vflip = 0):
    
    augmentation_operator = T.Compose([
        T.RandomEqualize(prob_equal),
        T.RandomInvert(prob_invert),
        T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation),
        T.RandomAffine(degrees = degree, translate = (hshift, vshift), scale = (1-scale, 1+scale), shear = (-shearx, shearx, -sheary, sheary)),
        T.RandomHorizontalFlip(prob_hflip),
        T.RandomVerticalFlip(prob_vflip),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return augmentation_operator
    
        
def get_no_augmentation():

    augmentation_operator = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return augmentation_operator
