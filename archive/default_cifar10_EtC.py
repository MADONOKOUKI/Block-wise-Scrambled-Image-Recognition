'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse

from no_adaptation_network import ShakePyramidNet
import tensorboardX as tbx
import re
from sklearn import preprocessing

from scheduler import CyclicLR
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import random
from util_norm import total_variation_norm
import math
from torch.optim import lr_scheduler

from etc_encryption import EtC_encryption
# For Help
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--milestones', default='150,225', type=str)

# For Networks
parser.add_argument("--depth", type=int, default=26)
parser.add_argument("--w_base", type=int, default=64)
parser.add_argument("--cardinary", type=int, default=4)

# For Training
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--nesterov", type=bool, default=True)
parser.add_argument('--e', '-e', default=150, type=int, help='learning rate')
parser.add_argument("--batch_size", type=int, default=128)

# file name
parser.add_argument("--tensorboard_name", type=str, default="proposed_adaptation_network_cifar100", help="tensorboard_name")
parser.add_argument("--training_model_name", type=str, default="proposed_adaptation_network_cifar100.t7", help="tensorboard_name")
parser.add_argument("--json_file_name", type=str, default="proposed_adaptation_network_cifar100.json", help="json_file")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
temp = 1
temp_min = 0.001
ANNEAL_RATE = 0.003

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# from cifar10 import CIFAR10
trainset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=16)

random.seed(30)
_shf = []
for i in range(64):
  _shf.append(i)
random.shuffle(_shf)

# Model
print('==> Building model..')

net = ShakePyramidNet(depth=110, alpha=270, label=10)
net = net.to(device)

if device == 'cuda':
    print("true")
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./'+args.training_model_name)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                lr=args.lr,
                momentum=0.9, 
                weight_decay=args.weight_decay,
                nesterov=args.nesterov)


l2_crit = nn.L1Loss()
mse = nn.MSELoss()

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    p = None
    param12 = 0.001
    param3 = 0.001
    param4 = 1e-1
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        x_stack = None
        imgs = inputs.numpy().astype('float32')

        inputs = EtC_encryption(imgs)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        true_loss = criterion(outputs, targets)

        loss = true_loss 
        loss.backward()
        optimizer.step()

        train_loss += true_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss, 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            x_stack = None
            imgs = inputs.numpy().astype('float32')

            inputs = EtC_encryption(imgs)

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = net(inputs)
            true_loss = criterion(outputs, targets)

            test_loss += true_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
       
    # Save checkpoint.
    acc = 100.*correct/total
    if best_acc < acc:
        # print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,'./'+args.training_model_name)
        best_acc = acc
    return test_loss, 100.*correct/total

writer = tbx.SummaryWriter(args.tensorboard_name)
scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')])

for epoch in range(start_epoch, start_epoch+args.e):
    scheduler.step()
    train_loss, train_acc = train(epoch+1)
    test_loss, test_acc = test(epoch+1)
    writer.add_scalars('data/loss',
    {
        'train_loss': train_loss / ((50000/512)+1),
        'test_loss': test_loss / ((10000/512)+1),
    },
        (epoch + 1)
    )
    writer.add_scalars('data/acc',
    {
        'train_acc': train_acc,
        'test_acc': test_acc
    },
        (epoch + 1)
    )
    print(str(train_loss / ((50000/512)+1)) +","+str(train_acc)+","+str(test_loss / ((10000/512)+1))+","+str(test_acc)+","+str(scheduler.get_lr()[0]))

print("best acc : ",best_acc)
writer.export_scalars_to_json("./"+args.json_file_name)
writer.close()
