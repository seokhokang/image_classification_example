import sys
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split

from torchvision import datasets as D
from torchvision import transforms as T

from model import CNN, training, inference

from sklearn.metrics import accuracy_score


dataset = sys.argv[1]
use_trained = False
model_path = './model_%s.pt'%dataset
frac_val = 0.1
batch_size = 64
seed = 42


## transform
if dataset in ['cifar10', 'svhn']:
    transform = T.Compose([
        T.ToTensor()
    ])

elif dataset in ['fashionmnist', 'mnist']:
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: torch.cat([x, x, x], 0)), 
        T.Resize((32, 32))
    ])
    
else:
    raise Exception


## data
if dataset == 'cifar10':
    trnval_set = D.CIFAR10(root='./data/cifar10', train=True, download=True, transform = transform)
    tst_set = D.CIFAR10(root='./data/cifar10', train=False, download=True, transform = transform)
    aug_params = {
        'degree': 30,
        'hshift': 0.3,
        'vshift': 0.3,
        'scale': 0.2,
        'shear': 15,
        'prob_hflip': 0.5,
        'prob_vflip': 0
    }

elif dataset == 'svhn':
    trnval_set = D.SVHN(root='./data/svhn', split='train', download=True, transform = transform)
    tst_set = D.SVHN(root='./data/svhn', split='test', download=True, transform = transform)
    trnval_set.targets = trnval_set.labels
    tst_set.targets = tst_set.labels
    aug_params = {
        'degree': 30,
        'hshift': 0.3,
        'vshift': 0.3,
        'scale': 0.2,
        'shear': 15,
        'prob_hflip': 0,
        'prob_vflip': 0
    }

elif dataset == 'fashionmnist':         
    trnval_set = D.FashionMNIST(root='./data/fashionmnist', train=True, download=True, transform = transform)
    tst_set = D.FashionMNIST(root='./data/fashionmnist', train=False, download=True, transform = transform)
    aug_params = {
        'degree': 30,
        'hshift': 0.3,
        'vshift': 0.3,
        'scale': 0.2,
        'shear': 15,
        'prob_hflip': 0.5,
        'prob_vflip': 0
    }

elif dataset == 'mnist':            
    trnval_set = D.MNIST(root='./data/mnist', train=True, download=True, transform = transform)
    tst_set = D.MNIST(root='./data/mnist', train=False, download=True, transform = transform)
    aug_params = {
        'degree': 30,
        'hshift': 0.3,
        'vshift': 0.3,
        'scale': 0.2,
        'shear': 15,
        'prob_hflip': 0,
        'prob_vflip': 0
    }

else:
    raise Exception


classes = np.unique(trnval_set.targets)
val_size = int(frac_val * len(trnval_set))
trn_size = len(trnval_set) - val_size
trn_set, val_set = random_split(trnval_set, [trn_size, val_size], generator=torch.Generator().manual_seed(seed))

print('trn/val/tst = %d/%d/%d'%(len(trn_set), len(val_set), len(tst_set)))

trn_loader = DataLoader(dataset=trn_set, batch_size=batch_size, shuffle=True, drop_last=True)  
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
tst_loader = DataLoader(dataset=tst_set, batch_size=batch_size, shuffle=False)


## training
cuda = torch.device('cuda:0')
net = CNN(len(classes)).to(cuda)
if use_trained == True:
    net.load_state_dict(torch.load(model_path))
else:
    net = training(net, trn_loader, val_loader, aug_params, model_path, cuda)

## test
tst_y = np.array(tst_set.targets)
tst_y_score, tst_y_hat = inference(net, tst_loader, cuda)

tst_acc = accuracy_score(tst_y, tst_y_hat)

print('--- test, processed %d, acc %6.4f'%(len(tst_set), tst_acc))
