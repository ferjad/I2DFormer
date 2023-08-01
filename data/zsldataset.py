#external libs
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import random
from os.path import join as ospj
from glob import glob 
#torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import pickle
#local imports
from data.common import get_norm_values
from clip.clip import tokenize
from clip import tokenize
import pymatreader
from utils.augmentations import ColorJitter, Lighting

def file_to_list(filename):
    with open(filename, 'r') as f:
        things = f.readlines()
    output = []
    for a in things:
        output.append(a.strip())
    return output

def dataset_transform(phase, norm_family = 'imagenet'):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use
        
        Returns
            transform: A list of pytorch transforms
    '''
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            jittering,
            lighting,
            transforms.Normalize(mean, std)
        ])

    elif phase == 'test_seen' or phase == 'test_unseen':
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform

class ZSL_dataset(Dataset):
    def __init__(
        self, 
        cfg, 
        phase,
        replace = None,
    ):
        self.cfg = cfg
        self.phase = phase
        self.transform = dataset_transform(self.phase)
        self.replace = replace

        self.split = pymatreader.read_mat(cfg.dataset.split_file)
        if self.cfg.dataset.pretrained_feats:
            with open(cfg.dataset.feats_file, 'rb') as f:
                self.feats = pickle.load(f)
        self.files = pymatreader.read_mat(cfg.dataset.filelist)['image_files']

        with open(cfg.dataset.articles, 'rb') as f:
            articles = pickle.load(f)

        self.articles = {}
        for bird, article in articles:
            self.articles[bird] = article

        if phase == 'train':
            key = ['trainval_loc']
            if cfg.dataset.mode == 'zsl':
                key.append('test_seen_loc')
            class_file = cfg.dataset.train_split
        else:
            if phase == 'test_unseen':
                key = ['test_unseen_loc']
                class_file = cfg.dataset.eval_split
            elif phase == 'test_seen':
                key = ['test_seen_loc']
                class_file = cfg.dataset.train_split
            
        all_class_file = ospj('/'.join(cfg.dataset.train_split.split('/')[:-1]), 'allclasses.txt')

        phase_classes = file_to_list(class_file)

        all_classes = file_to_list(all_class_file)

        print(f'Key is {key}')
        self.class_names = []
        self.class_to_label = {}
        self.samples = []
        
        
        ignore_classes = []
        class_count = 0
        if phase == 'train':
            for idx, class_name in enumerate(phase_classes):
                if class_name in ignore_classes:
                    print(f'Skipping {class_name}')
                    continue
                self.class_names.append(class_name)
                self.class_to_label[class_name] = class_count
                class_count += 1
        else:
            mask = []
            for idx, class_name in enumerate(all_classes):
                self.class_names.append(class_name)
                self.class_to_label[class_name] = class_count
                class_count += 1
    
                if class_name in phase_classes:
                    mask.append(True)
                else:
                    mask.append(False)
            mask = torch.Tensor(mask).bool()
            if phase == 'test_seen':
                print(f'Inverting mask for test seen')
                mask = ~mask
            self.unseen_mask = mask


        for k in key:
            for idx in self.split[k]:
                idx = idx - 1
                sample = '/'.join(self.files[idx].split('/')[-2:]).strip() # removing trailing space
                sample = ospj(cfg.dataset.root, sample)
                class_name = sample.split('/')[-2]
                if self.cfg.dataset.pretrained_feats:
                    sample = idx
                self.samples.append([class_name, sample])
        

    def __getitem__(self, index):
        class_name, sample = self.samples[index]
        if self.cfg.dataset.pretrained_feats:
            image = self.feats['features'][sample, :]
        else:
            image = Image.open(sample).convert('RGB')
            image = self.transform(image)

        label = self.class_to_label[class_name]
        return image, label

    def __len__(self):
        return len(self.samples)