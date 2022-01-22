import torch
from torch import nn

from torchvision import transforms as T


def get_trn_augmentation(degree = 30, hshift = 0.3, vshift = 0.3, scale = 0.2, brightness = 0.2, contrast = 0.2, prob_hflip = 0.5, prob_vflip = 0):
    
    augmentation_operator = T.Compose([
        T.RandomAffine(degrees = degree, translate = (hshift, vshift), scale = (1-scale, 1+scale)),
        T.ColorJitter(brightness = brightness, contrast = contrast),
        T.RandomHorizontalFlip(prob_hflip),
        T.RandomVerticalFlip(prob_vflip),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return augmentation_operator
    

def get_class_adaptive_trn_augmentation(degree = [30] * 10, hshift = [0.3] * 10, vshift = [0.3] * 10, scale = [0.2] * 10, brightness = [0.2] * 10, contrast = [0.2] * 10, prob_hflip = [0.5] * 10, prob_vflip = [0] * 10):
    
    augmentation_operator = {}
    n_class = len(degree)
    for i in range(n_class):
        augmentation_operator[i] = T.Compose([
            T.RandomAffine(degrees = degree[i], translate = (hshift[i], vshift[i]), scale = (1-scale[i], 1+scale[i])),
            T.ColorJitter(brightness = brightness[i], contrast = contrast[i]),
            T.RandomHorizontalFlip(prob_hflip[i]),
            T.RandomVerticalFlip(prob_vflip[i]),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return augmentation_operator
    
        
def get_tst_augmentation():

    augmentation_operator = T.Compose([
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return augmentation_operator