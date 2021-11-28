import torch
import torch.optim as optim
from .scheduler import PolyLR, CosineWithRestarts
from data_utils.AdaBelief import AdaBelief

def create_optimizer(model, config):
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*config.lr},
        {'params': model.classifier.parameters(), 'lr': config.lr},
        ], lr=config.lr, betas=(0.9, 0.999), eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'adabelief':
        optimizer = AdaBelief(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*config.lr},
        {'params': model.classifier.parameters(), 'lr': config.lr},
        ], lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify=False)
    
    if config.scheduler_type=='polylr':
        scheduler = PolyLR(optimizer, config.total_itrs, power=0.9)
    elif config.scheduler_type=='cosine_restart':
        scheduler = CosineWithRestarts(optimizer, t_max=10)
    else:
        print('no scheduler set')
        #raise NotImplementedError("indicate scheduler type in cfg.py")
        
    return optimizer, scheduler

