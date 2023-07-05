import os
import cv2
from torch.utils.data import Dataset
import copy
import random
import numpy as np
from functions import *
import torchvision.transforms as transforms

class CifarDataset(Dataset):
    def __init__(self, opt, phase = 'train'):
        self.opt = opt
        self.phase = phase
        self.dir_data = os.path.join(opt.dir_data, phase)
        self.obj_list = []
        self.label_list = []
        
        self.init_db()
        
    def init_db(self):
        for idx, model_category in enumerate(sorted(os.listdir(self.dir_data))):
            for model_id in sorted(os.listdir(os.path.join(self.dir_data, model_category))):
                self.obj_list.append(os.path.join(self.dir_data, model_category, model_id))
                self.label_list.append(idx)
                
    def __len__(self):
        return len(self.obj_list)
    
    def __getitem__(self, idx):
        dir_path = self.obj_list[idx]
        image = np.array(augment_image(pil_loader(dir_path)))
        label = self.label_list[idx]
        
        return image, label
    
    def sample(self, sample_size):
        idx_list = random.sample(range(len(self.obj_list)), sample_size)
        image_list = []
        label_list = []
        for idx in idx_list:
            image, label = self.__getitem__(idx)
            image_list.append(image)
            label_list.append(label)
        # return torch.stack(image_list, dim=0), torch.Tensor(label_list)
        return image_list, label_list
        
        
