""" Contains helper functions forprinting, training, validating, testing NN models."""

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
from utils import *

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.epoch_sum = 0
        self.epoch_count = 0
        self.epoch_avg = 0

    def reset(self):
#         self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.epoch_sum += val * n
        self.epoch_count += n
        self.epoch_avg = self.epoch_sum / self.epoch_count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '} ({epoch_avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, scheduler, epoch, **kwargs):
    device = kwargs['device']
    batch_print_freq = kwargs['batch_print_freq']
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    train_bg_losses = []
    train_bg_acc = []

    # switch to train mode
    if scheduler:
        scheduler.step()
    model.train()

    startep = end = time.time()
    for i, (images, target) in enumerate(train_loader, 1):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device) #, non_blocking=True)
        target = target.to(device) #, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % batch_print_freq == 0:
            progress.display(i)
            train_bg_losses.append(losses.avg)
            train_bg_acc.append(top1.avg)
            batch_time.reset()
            data_time.reset()
            losses.reset()
            top1.reset()
            top5.reset() 
    
        end = time.time()

    eptime = time.time() - startep
    
    if len(train_loader) % batch_print_freq != 0:
        progress.display(i)
    print(' * TRAIN: Time {eptime:6.3f} Acc@1 {top1.epoch_avg:.3f} Acc@5 {top5.epoch_avg:.3f}'.format(eptime=eptime, top1=top1, top5=top5))
  
    return top1.epoch_avg, losses.epoch_avg, train_bg_acc, train_bg_losses, eptime


def validate(val_loader, model, criterion, **kwargs):
    device = kwargs['device']
    batch_print_freq = kwargs['batch_print_freq']
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Val: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader, 1):
            images = images.to(device) #, non_blocking=True)
            target = target.to(device) #, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % batch_print_freq == 0:
                progress.display(i)
                batch_time.reset()
                losses.reset()
                top1.reset()
                top5.reset()

        if len(val_loader) % batch_print_freq != 0:
            progress.display(i)
  
    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.epoch_avg:.3f} Acc@5 {top5.epoch_avg:.3f}'.format(top1=top1, top5=top5))
  
    return top1.epoch_avg, losses.epoch_avg


def test(model, dataloader, criterion, **kwargs):
    device = kwargs['device']
    batch_print_freq = kwargs['batch_print_freq']

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(dataloader, 1):
            images = images.to(device) #, non_blocking=True)
            target = target.to(device) #, non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % batch_print_freq == 0:
                progress.display(i)
                batch_time.reset()
                top1.reset()
                top5.reset()

        if len(dataloader) % batch_print_freq != 0:
            progress.display(i)
  
    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.epoch_avg:.3f} Acc@5 {top5.epoch_avg:.3f}'.format(top1=top1, top5=top5))

    return top1.epoch_avg


def train_model(model, dataloaders, criterion, optimizer, lrscheduler, epochs=1, start_epoch=0, **kwargs):
  
    best_acc1 = 0

    ## Load Data
    trainloader, valloader = dataloaders['train'], dataloaders['val']

    stats = {}
    stats['train_bg_losses'] = []
    stats['train_bg_acc'] = []
    stats['train_ep_losses'] = []
    stats['train_ep_acc'] = []
    stats['train_ep_time'] = []
    stats['val_ep_losses'] = []
    stats['val_ep_acc'] = []
  
    # print("Before training starts: " + str(time.time() - start) + " ms")
    for epoch in range(start_epoch, epochs):

        # train for 1 epoch
        tr_acc1, tr_eploss, tr_bg_acc1, tr_bg_loss, tr_ep_time = train(trainloader, model, criterion, optimizer, lrscheduler, epoch,  **kwargs)

        stats['train_ep_acc'].append(tr_acc1)
        stats['train_ep_losses'].append(tr_eploss)
        stats['train_ep_time'].append(tr_ep_time)
        stats['train_bg_acc'].extend(tr_bg_acc1)
        stats['train_bg_losses'].extend(tr_bg_loss)

        # evaluate on validation set
        acc1, eploss = validate(valloader, model, criterion, **kwargs)

        stats['val_ep_acc'].append(acc1)
        stats['val_ep_losses'].append(eploss)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

#     # add if condition to save more sparsely than each epoch
#     if lrscheduler:
#       save_checkpoint({
#           'epoch': epoch + 1,
#           'arch': None, # architecture
#           'state_dict': model.state_dict(),
#           'best_acc1': best_acc1,
#           'optimizer': optimizer.state_dict(),
#           'lrscheduler': lrscheduler.state_dict()
#       }, is_best)

#     else:
#       save_checkpoint({
#           'epoch': epoch + 1,
#           'arch': None, # architecture
#           'state_dict': model.state_dict(),
#           'best_acc1': best_acc1,
#           'optimizer': optimizer.state_dict(),
#       }, is_best)

    return stats

