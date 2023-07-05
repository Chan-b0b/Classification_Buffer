from Opt import Options
from Dataset import CifarDataset
from Buffer import Buffer
from functions import * 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from model import *
from pathlib import Path

from tensorboardX import SummaryWriter

# def train_epoch(model, inputs, labels, phase):
#     inputs = torch.Tensor(inputs)
#     labels = torch.Tensor(labels)

#     inputs = inputs.to(device)
#     labels = labels.to(device)

#     scores = []
#     # zero the parameter gradients
#     optimizer.zero_grad()

#     # forward
#     # track history if only in train
#     with torch.set_grad_enabled(phase == 'train'):
#         outputs = model_ft(inputs)
        
#         _, preds = torch.max(outputs, 1)
#         loss = criterion(outputs, labels.long())

#         # backward + optimize only if in training phase
#         if phase == 'train':
#             loss.backward()
#             optimizer.step()
#     return loss

# def train_epoch(opt, model, images, labels, phase):
#     images = torch.Tensor(images)
#     scores = []
#     running_corrects = 0
#     with torch.set_grad_enabled(phase == 'train'):
#         for idx, image in enumerate(images):
#             optimizer.zero_grad()
#             image = torch.stack([augment_image(transforms.functional.to_pil_image(image), basic = False) for _ in range(opt.augment_num)])
#             image = image.to(opt.device)
#             outputs = model(image)
            
#             _, preds = torch.max(outputs, 1)
#             label = torch.tensor([labels[idx] for _ in range(len(outputs))]).to(opt.device)
#             loss = opt.criterion(outputs, label.long())
            
#             if opt.score_criterion == 'std':
#                 scores.append(outputs.std(dim=0).mean().item())
#             elif opt.score_criterion == 'error':
                
#                 scores.append(loss.item())            

#             if phase == 'train':
#                 loss.backward()
#                 optimizer.step()
#             running_corrects += torch.sum(preds == labels[idx])   
#     return scores, running_corrects.double() / (opt.augment_num * len(images))

def train_epoch(opt, model, images, labels, phase):
    images = torch.Tensor(images)
    scores = []
    running_corrects = 0
    with torch.set_grad_enabled(phase == 'train'):
        aug_images = []
        optimizer.zero_grad()
        for idx, image in enumerate(images):
            aug_images.extend([augment_image(transforms.functional.to_pil_image(image), basic = False) for _ in range(opt.augment_num)])
        aug_images = torch.stack(aug_images)
        aug_images = aug_images.to(opt.device)
        outputs = model(aug_images)
        
        _, preds = torch.max(outputs, 1)
        # label = torch.tensor([labels[idx] for _ in range(len(outputs))]).to(opt.device)
        label = torch.Tensor(labels).repeat_interleave(opt.augment_num).to(opt.device)
        loss = opt.criterion(outputs, label.long())
        
        if opt.score_metric == 'std':
            scores = outputs.std(dim=1).reshape(opt.augment_num,-1).mean(0).detach().cpu().numpy()
        elif opt.score_metric == 'error':
            
            scores = opt.score_criterion(outputs, label.long()).reshape(opt.augment_num, -1).mean(0).detach().cpu().numpy()           

        if phase == 'train':
            loss.backward()
            optimizer.step()
        running_corrects += torch.sum(preds == label).item()
    return scores, running_corrects / (opt.augment_num * len(images))

opt = Options().parse()

env_buffer = Buffer(opt)
init_dataset = CifarDataset(opt, phase='train')   
test_dataset = CifarDataset(opt, phase='test')

opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_writer = SummaryWriter(get_result_folder(opt.dir_checkpoints))

log_dir = get_result_folder(opt.dir_results)
Path(log_dir).mkdir(parents=True, exist_ok=True)

model_ft = models.wide_resnet101_2(pretrained=True)
num_ftrs = model_ft.fc.in_features

half_in_size = round(num_ftrs/2)
layer_width = 20 #Small for Resnet, large for VGG
Num_class=10
    
model_ft.fc = SpinalNet_ResNet(half_in_size, layer_width, Num_class) #SpinalNet_ResNet
model_ft = model_ft.to(opt.device)
    
optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
opt.criterion = nn.CrossEntropyLoss()
opt.score_criterion = nn.CrossEntropyLoss(reduction='none')
# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


while len(env_buffer) < opt.init_buffer_size:
    new_data = init_dataset.sample(opt.sample_size)
    env_buffer.input(model_ft, new_data, init=True)

for epoch in range(opt.epoch):
    inputs, labels = env_buffer.sample_replay_level(opt.batch_size)
    scores, accuracy = train_epoch(opt, model_ft, inputs, labels, phase='train')
    train_writer.add_scalar('Loss/train', np.mean(scores), epoch)
    train_writer.add_scalar('Accuracy/train', accuracy, epoch)

    env_buffer.update_seed(np.array(scores))
    
    new_data = init_dataset.sample(opt.sample_size)
    input_size = env_buffer.input(model_ft, new_data, init=False)
    train_writer.add_scalar('Input_size', input_size, epoch)

    test_data = test_dataset.sample(opt.test_size)
    images, labels = new_data
    test_scores, test_accuracy = train_epoch(opt, model_ft, np.array(images), labels, phase='test')
    train_writer.add_scalar('Loss/test', np.mean(test_scores), epoch)
    train_writer.add_scalar('Accuracy/test', test_accuracy, epoch)

    if epoch % 10 == 0:
        print('Epoch {}/{}'.format(epoch+1, opt.epoch))
        print('train_score: {:.4f} test_score: {:.4f} train_accuracy: {:.4f} test_accuracy: {:.4f}'.format(np.mean(scores), np.mean(test_scores), accuracy, test_accuracy))

        if epoch % 20 == 0:
            save_checkpoints(log_dir, model_ft, epoch)