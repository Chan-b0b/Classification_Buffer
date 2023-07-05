import numpy as np
import torch
import torch.nn as nn
from Dataset import CifarDataset
from functions import *
import torchvision.transforms as transforms


class Buffer:
    def __init__(self, 
        opt,
        rho=1.0, 
        alpha = 0.9,
        score_transform='power',
        temperature=1.0, 
        staleness_coef=0, 
        staleness_transform='power', 
        staleness_temperature=1.0, 
        ):
        self.opt = opt
        
        self.seed_buffer_size = opt.sample_size*40
        N = self.seed_buffer_size
        
        # assert N % opt.batch_size == 0
        
        self.images = np.zeros((N,3,256,256), dtype=np.float)
        self.labels = np.zeros(N, dtype = np.int32)
        self.score_transform = score_transform
        self.seed_scores = np.array([0.]*N, dtype=np.float)
        self.seed_staleness = np.array([0.]*N, dtype=np.float)
        self.unseen_seed_weights = np.array([1.]*N)
        self.alpha = alpha
        self.running_sample_count = 0
        self.temperature = temperature
        self.rho = rho
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        
        
        self.loader = pil_loader
        
        self.augment_num = opt.augment_num    
        self.criterion = nn.CrossEntropyLoss()     
         
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                 
    def __len__(self):
        return int(self.seed_buffer_size - (self.unseen_seed_weights > 0).sum())
    
    def input(self, model, new_data, init = False):
        images, labels = new_data
        # images = torch.Tensor(np.array(images))
        scores = np.ones_like(labels, dtype=np.float32)
        if not init:
            with torch.no_grad():
                for idx, image in enumerate(images):
                    image = torch.Tensor(image)
                    image = torch.stack([augment_image(transforms.functional.to_pil_image(image), basic = False) for _ in range(self.augment_num)])
                    image = image.to(self.device)
                    outputs = model(image)
                    
                    if self.opt.score_metric == 'std':
                        scores[idx] = outputs.std(dim=0).mean().item()
                    elif self.opt.score_metric == 'error':
                        # _, preds = torch.max(outputs, 1)
                        label = torch.tensor([labels[idx] for _ in range(len(outputs))])
                        scores[idx] = self.criterion(outputs.cpu(), label)
            
            # aug_images = []
            # for idx, image in enumerate(images):
            #     image = torch.Tensor(image)
            #     aug_images.extend([augment_image(transforms.functional.to_pil_image(image), basic = False) for _ in range(self.augment_num)])
            # aug_images = torch.stack(aug_images)
        images = self.to_numpy(images)
        labels = self.to_numpy(labels)
        scores = self.to_numpy(scores)
        
        threshold = self.seed_scores.min()
        sampled = scores > threshold
        images = images[sampled]
        labels = labels[sampled]
        scores = scores[sampled]
        batch_size = len(images)
        
        if batch_size != 0 :
            if self.__len__() == self.seed_buffer_size:
                new_index = np.argpartition(self.seed_scores,batch_size)[:batch_size]
            else:
                new_index = np.where(self.seed_scores == 0)[0][:batch_size]              
            self.images[new_index] = images
            self.labels[new_index] = labels
            self.seed_scores[new_index] = scores
            self.seed_staleness[new_index] = 0
            self.unseen_seed_weights[new_index] = 0
        return batch_size    
    def update_seed(self, score, update_staleness=True):
           
        if update_staleness:
            self._update_staleness(self.seed_idx)   
            
        old_scores = self.seed_scores[self.seed_idx]
        self.seed_scores[self.seed_idx] = (1 - self.alpha)*old_scores + self.alpha*score    
        
    def sample_replay_level(self, sample_num, update_staleness=True):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=np.float)/len(sample_weights)
            sample_weights = sample_weights*(1-self.unseen_seed_weights)
            sample_weights /= np.sum(sample_weights)
            
        elif np.sum(sample_weights, 0) != 1.0:
            sample_weights = sample_weights/np.sum(sample_weights,0)
            
        self.seed_idx = np.random.choice(range(len(sample_weights)), size=sample_num, replace=False, p=sample_weights)
        
        if update_staleness:
            self._update_staleness(self.seed_idx)       
             
        images = self.images[self.seed_idx]
        labels = self.labels[self.seed_idx]
        
        return images, labels
    
    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        weights = weights * (1-self.unseen_seed_weights) # zero out unseen levels

        z = np.sum(weights)
        if z > 0:
            weights /= z
        else:
            weights = np.ones_like(weights, dtype=np.float)/len(weights)
            weights = weights * (1-self.unseen_seed_weights)
            weights /= np.sum(weights)

        staleness_weights = 0
        
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(self.staleness_transform, self.staleness_temperature, self.seed_staleness)
            staleness_weights = staleness_weights * (1-self.unseen_seed_weights)
            z = np.sum(staleness_weights)
            if z > 0: 
                staleness_weights /= z
            else:
                staleness_weights = 1./len(staleness_weights)*(1-self.unseen_seed_weights)

            weights = (1 - self.staleness_coef)*weights + self.staleness_coef*staleness_weights

        return weights
    
    def _score_transform(self, transform, temperature, scores):
        if transform == 'constant':
            weights = np.ones_like(scores)
        if transform == 'max':
            weights = np.zeros_like(scores)
            scores = scores[:]
            scores[self.unseen_seed_weights > 0] = -float('inf') # only argmax over seen levels
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            weights[argmax] = 1.
        elif transform == 'eps_greedy':
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1. - self.eps
            weights += self.eps/len(self.seeds)
        elif transform == 'rank':
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores).clip(0) + eps) ** (1./temperature)
        elif transform == 'softmax':
            weights = np.exp(np.array(scores)/temperature)
        elif transform == 'match':
            weights = np.array([(1-score)*score for score in scores])
            weights = weights ** (1./temperature)
        elif transform == 'match_rank':
            weights = np.array([(1-score)*score for score in scores])
            temp = np.flip(weights.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)

        return weights
    
    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.seed_staleness = self.seed_staleness + 1
            self.seed_staleness[selected_idx] = 0
            
    def to_numpy(self, input):
        
        if type(input) == torch.Tensor:
            input = input.detach().cpu()
        return np.array(input)