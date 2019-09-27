"""
1. Clone repository
2. Provide ImageNet Path on line 39
3. Run script with Python3.6 (that's the one I ran with):
   $ nvprof --profile-from-start off --print-gpu-trace --print-api-trace --csv --normalized-time-unit us --log-file /path/to/log.csv python3.6 nvtx_profile.py

Docker Image: nightly-devel-cuda10.0-cudnn7
"""
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


def main():    
    model = models.resnet152(pretrained=False)

    ## Hyper-parametes
    batch_size = 80

    ## Misc configs
    print_freq = 1  # 500
    
    ## DataLoaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformation = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,])
    trainset = torchvision.datasets.ImageFolder('/workspace/data/train', transform=transformation) # ImageNet path
    # trainset = torchvision.datasets.FakeData(size=6500, image_size=(3,224,224), num_classes=1000, transform=transformation['val'])
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=1)

    ## Model, Optimizer, Scheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # lrscheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ## Training
    model.train()

    trainiter = iter(trainloader)
    batches = 5
    total = 0
    correct = 0
    running_loss = 0.0

    # with torch.autograd.profiler.profile(enabled=False, use_cuda=True, record_shapes=True) as prof:
    with torch.cuda.profiler.profile():
        '''# warm up CUDA memory allocator and profiler
        with torch.no_grad():
            dummyiter = iter(trainloader)
            dummydata, _ = dummyiter.next()
            dummydata = dummydata.to(device)
            model(dummydata)'''
        
        # profiling now
        with torch.autograd.profiler.emit_nvtx(record_shapes=True):
            for i in range(batches):
                images, target = trainiter.next()
                images = images.to(device)
                target = target.to(device)

                output = model(images)
                loss = criterion(output, target)
                # compute gradients and Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # lrscheduler.step()

                # loss and accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                running_loss += loss.item()
                if (i+1) % print_freq == 0:
                    print(" * Train [{}/{}]: Accuracy: {:.3f}\tLoss: {:.5f}".format(i+1, batches, (correct/total), running_loss/print_freq))
                    correct = total = 0
                    running_loss = 0.0

if __name__ == '__main__':
    main()

