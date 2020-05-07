# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:34:37 2020

@author: Rafael Fricks

Rerun the evaluation for the pretrained network
"""

import torch
from eval_model import *
from torchvision import datasets, models, transforms
import torch.nn as nn


PATH_TO_IMAGES = '/data/'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

N_LABELS = 14 

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
    # add final layer with # outputs in same dimension of labels with sigmoid
    # activation
model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    # put model on GPU
model = model.cuda()

# w = torch.load('results/checkpoint')
# # print(w['model'])
# model.load_state_dict(w['model'])

checkpoint_best = torch.load('results/checkpoint')
model = checkpoint_best['model']


data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

print('everything defined')
make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES)
