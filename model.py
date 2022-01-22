import numpy as np
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.models import vgg16_bn
from torchvision import transforms as T

from scipy.special import softmax
from sklearn.metrics import accuracy_score, log_loss

from utils import get_trn_augmentation, get_class_adaptive_trn_augmentation, get_tst_augmentation


class CNN(nn.Module):

    def __init__(self, n_classes):
        
        super().__init__()
        self.features = vgg16_bn(pretrained = False).features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, n_classes)
             
    def forward(self, x):
            
        x = self.features(x)
        x = self.gap(x).squeeze(3).squeeze(2)
        x = self.classifier(x)

        return x

    
def training(net, trn_loader, val_loader, model_path, cuda, max_epochs = 500, patience = 20):
    
    #trn_augmentation = get_trn_augmentation()
    trn_augmentation = get_class_adaptive_trn_augmentation()
    
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6, verbose=True)

    val_y = np.array(val_loader.dataset.dataset.targets)[val_loader.dataset.indices]
    val_log = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(trn_loader):
    
            batch_x, batch_y = batchdata
            batch_x, batch_y = batch_x.to(cuda), batch_y.to(cuda)
            
            #batch_x = torch.stack([trn_augmentation(x) for x in batch_x])
            batch_x = torch.stack([trn_augmentation[y.item()](x) for x, y in zip(batch_x, batch_y)])

            batch_y_hat = net(batch_x)
    
            loss = loss_fn(batch_y_hat, batch_y).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            trn_loss = loss.detach().item()
    
        print('--- training epoch %d, processed %d/%d, loss %.4f, time elapsed(min) %.2f' %(epoch, len(trn_loader.dataset), len(trn_loader.dataset), trn_loss, (time.time()-start_time)/60))
    
        # validation
        val_y_score, val_y_hat = inference(net, val_loader, cuda)

        val_loss = log_loss(val_y, softmax(val_y_score, 1), labels = list(range(val_y_score.shape[1])))
        val_acc = accuracy_score(val_y, val_y_hat)
    
        val_log[epoch] = val_loss
        print('--- validation epoch %d, processed %d, current loss %.4f, acc %.4f, best loss %.4f, time elapsed(min) %.2f' %(epoch, len(val_loader.dataset), val_loss, val_acc, np.min(val_log[:epoch + 1]), (time.time()-start_time)/60))
    
        lr_scheduler.step(val_loss)
    
        # earlystopping
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        elif np.argmin(val_log[:epoch + 1]) <= epoch - patience:
            break
    
    print('training terminated at epoch %d' %epoch)
    net.load_state_dict(torch.load(model_path))

  
def inference(net, tst_loader, cuda):

    tst_augmentation = get_tst_augmentation()
    
    net.eval()

    tst_y_score = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(tst_loader):
            
            batch_x, batch_y = batchdata
            batch_x = batch_x.to(cuda)
            batch_y = batch_y.numpy()
            
            batch_x = torch.stack([tst_augmentation(x) for x in batch_x])

            batch_y_score = net(batch_x).cpu().numpy()
    
            tst_y_score.append(batch_y_score)
    
    tst_y_score = np.vstack(tst_y_score)
    tst_y_hat = np.argmax(tst_y_score, 1)
    
    return tst_y_score, tst_y_hat