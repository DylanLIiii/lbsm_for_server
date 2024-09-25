# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

def calculate_z_mask(outputs, targets, larger=True):
    """
    Calculate the mean of outputs smaller than the output of the target class.

    Args:
        outputs (torch.Tensor): Model outputs of shape (batch_size, num_classes).
        targets (torch.Tensor): Target labels of shape (batch_size,).

    Returns:
        torch.Tensor: Mean of outputs smaller than the target class output, shape (batch_size, 1).
    """
    sorted_outputs, _ = outputs.sort(dim=-1, descending=True)
    zn = torch.gather(outputs, -1, targets.unsqueeze(1).long())
    if not larger:
        mask = sorted_outputs < zn
    else:
        mask = sorted_outputs > zn
    sorted_outputs[~mask] = 0  # Keep only masked values
    
    non_zero_mask = sorted_outputs != 0
    non_zero_sum = torch.sum(sorted_outputs, dim=-1, keepdim=True)
    non_zero_count = torch.sum(non_zero_mask.float(), dim=-1, keepdim=True)
    
    z_mask = non_zero_sum / (non_zero_count + 1e-8)  # Add small epsilon to avoid division by zero?
    return z_mask

# Experiment1 : No mixup and cutmix, label smoothing | z_top1 - zn | increasing smoothing from 0.1 - 0.2 
def train_one_epoch1(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     # comment this line to disable mixup, cutmix and label smoothing
        #     #samples, mixed_targets, lam = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            zn = torch.gather(outputs, 1, targets.unsqueeze(-1).long())
            z_top1, _ = outputs.topk(1, dim=-1)
            reg = z_top1 - zn 
            smoothing = 0.1 + 0.1 * epoch / 299
            one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=outputs.size(1)).float()
            loss = criterion(samples, outputs, one_hot_targets) + smoothing * reg.mean()
        
            

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Experiment2 : No mixup and cutmix, label smoothing | z_larger - zn | increasing smoothing from 0.1 - 0.2 
def train_one_epoch2(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     # comment this line to disable mixup, cutmix and label smoothing
        #     #samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            smoothing = 0.1 + 0.1 * epoch / 299
            
            zn = torch.gather(outputs, 1, targets.unsqueeze(-1).long())
            z_larger = calculate_z_mask(outputs, targets, larger=True)
            reg = z_larger - zn 
            one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=outputs.size(1)).float()
            loss = criterion(samples, outputs, one_hot_targets) + smoothing * reg.mean()
            

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Experiment3: Enable mixup and cutmix, disable label smoothing, | z_smaller + top2zc - top2zn | increasing smoothing from 0.1 - 0.2 
def train_one_epoch3(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            # comment this line to disable mixup, cutmix and label smoothing
            # should change mixup_fn to return lam 
            # or calculate by targets 
            samples, mixed_targets = mixup_fn(samples, targets)
            
                   
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            # remember to set label smoothing to 0
            outputs = model(samples)
            smoothing = 0.1 + 0.1 * epoch / 299
            two_targets, two_indices = mixed_targets.topk(2, dim=-1) 
            target1_lam = two_targets[:, 0] / (1. - args.smoothing + args.smoothing / 1000)
            target2_lam = two_targets[:, 1] / (1. - args.smoothing + args.smoothing / 1000)
            target1 = two_indices[:, 0]
            target2 = two_indices[:, 1]
            
            # consider in two target 
            zn1 = torch.gather(outputs, -1, target1.view(-1,1).long())
            z_smaller1 = calculate_z_mask(outputs, target1, larger=False)
            
            zn2 = torch.gather(outputs, -1, target2.view(-1,1).long())
            z_smaller2 = calculate_z_mask(outputs, target2, larger=False)
            
            reg_smaller1 = zn1 - z_smaller1 
            reg_smaller2 = zn2 - z_smaller2
            
            z_top2, _ = outputs.topk(2, dim=-1)
            z_top2 = z_top2[:,1]
            reg1 = z_top2 - zn1
            reg2 = z_top2 - zn2 
            
            weighted_reg_smaller = target1_lam.mean() * reg_smaller1 + target2_lam.mean() * reg_smaller2
            non_weighted_reg = 0.5 * reg1 + 0.5 * reg2
            
            loss = criterion(samples, outputs, mixed_targets) + smoothing * (weighted_reg_smaller.mean() + non_weighted_reg.mean())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# Experiment4: Enabel mixup and Cutmix, Enable label smoothing, increasing smoothing from 0.1 - 0.2 
def train_one_epoch4(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            # comment this line to disable mixup, cutmix and label smoothing
            # should change mixup_fn to return lam 
            # or calculate by targets  
            samples, mixed_targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            # remember to set label smoothing to 0
            outputs = model(samples)
            smoothing = 0.1 + 0.1 * epoch / 299
            two_targets, two_indices = mixed_targets.topk(2, dim=-1) 
            target1_lam = two_targets[:, 0] / (1. - args.smoothing + args.smoothing / 1000)
            target2_lam = two_targets[:, 1] / (1. - args.smoothing + args.smoothing / 1000)
            target1 = two_indices[:, 0]
            target2 = two_indices[:, 1]
            # first solution
            zn1 = torch.gather(outputs, -1, target1)
            zn2 = torch.gather(outputs, -1, target2)
            reg1 = zn1 - outputs.mean(dim=-1, keepdim=True)
            reg2 = zn2 - outputs.mean(dim=-1, keepdim=True)
            loss1 = target1_lam.mean() * (criterion(samples, outputs, target1.view(-1,1)) + smoothing * reg1.mean())
            loss2 = target2_lam.mean() * (criterion(samples, outputs, target2.view(-1,1)) + smoothing * reg2.mean())
            loss = loss1 + loss2
            
            # second solution 
            # ls_criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
            # loss = ls_criterion(outputs, mixed_targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Experiment5: Enabel mixup and Cutmix, | smaller + (-larger), increasing smoothing from 0.1 - 0.2 
def train_one_epoch5(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            # comment this line to disable mixup, cutmix and label smoothing
            # should change mixup_fn to return lam 
            # or calculate by targets  
            samples, mixed_targets, = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            # remember to set label smoothing to 0
            outputs = model(samples)
            smoothing = 0.1 + 0.1 * epoch / 299
            two_targets, two_indices = mixed_targets.topk(2, dim=-1) 
            target1_lam = two_targets[:, 0] / (1. - args.smoothing + args.smoothing / 1000)
            target2_lam = two_targets[:, 1] / (1. - args.smoothing + args.smoothing / 1000)
            target1 = two_indices[:, 0]
            target2 = two_indices[:, 1]
            zn1 = torch.gather(outputs, -1, target1.view(-1,1).long())
            z_smaller1 = calculate_z_mask(outputs, target1, larger=False)
            z_larger1 = calculate_z_mask(outputs, target1, larger=True)
            zn2 = torch.gather(outputs, -1, target2.view(-1,1).long())
            z_smaller2 = calculate_z_mask(outputs, target2, larger=False)
            z_larger2 = calculate_z_mask(outputs, target2, larger=True)
            
            reg_smaller1 = zn1 - z_smaller1 
            reg_larger1 = z_larger1 - zn1
            reg_smaller2 = zn2 - z_smaller2 
            reg_larger2 = z_larger2 - zn2
            
            one_hot_targets1 = torch.nn.functional.one_hot(target1, num_classes=outputs.size(1)).float()
            one_hot_targets2 = torch.nn.functional.one_hot(target2, num_classes=outputs.size(1)).float()
            loss1 = criterion(samples, outputs, one_hot_targets1) + smoothing * (reg_smaller1.mean() + reg_larger1.mean())
            loss2 = criterion(samples, outputs, one_hot_targets2) + smoothing * (reg_smaller2.mean() + reg_larger2.mean())
            loss = target1_lam.mean() * loss1 + target2_lam.mean() * loss2
            
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
