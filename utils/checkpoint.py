import os

import mmcv
from mmcv.runner import weights_to_cpu

import torch
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer


def get_opt_state_dict(optimizer):
    return optimizer.state_dict()


def save_checkpoint(model, filepath, optimizer, scaler=None, scheduler=None, anomaly_info=None, meta=None):
    checkpoint = dict(meta=meta if meta is not None else dict(),
                      anomaly_info=anomaly_info if anomaly_info is not None else dict(),
                      state_dict=weights_to_cpu(model.state_dict()))
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = get_opt_state_dict(optimizer)
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = dict()
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = get_opt_state_dict(optim)
        
    if scaler is not None:
        if isinstance(scaler, GradScaler):
            checkpoint['scaler'] = scaler.state_dict()
        elif isinstance(optimizer, dict):
            checkpoint['scaler'] = dict()
            for name, optim in optimizer.items():
                checkpoint['scaler'][name] = optim.state_dict()
    else:
        checkpoint['scaler'] = dict()
        
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    else:
        checkpoint['scheduler'] = dict()

    mmcv.mkdir_or_exist(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()


def resume(args, model, optimizer, scaler=None, scheduler=None):
    checkpoint = torch.load(args.resume_from)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
        
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    args.start_epoch = checkpoint['meta']['epoch']

    return


