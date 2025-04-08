import os
import pdb
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms


# LOCATION = "/gscratch/krishna/gstoica3/datasets/"
LOCATION = "/weka/prior-default/georges/research/MergedVisionEncoders/datasets/"

class OxfordPets(Dataset):
    """`Oxford Pets <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'pets'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 loader=default_loader,
                 base_set=None):
        
        self.root_og = root
        self.name = 'pets'
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.train = train
        self.transform = transform
        self.loader = loader
        self.classnames = [
            'Abyssinian',
            'american bulldog',
            'american pit bull terrier',
            'basset hound',
            'beagle',
            'Bengal',
            'Birman',
            'Bombay',
            'boxer',
            'British Shorthair',
            'chihuahua',
            'Egyptian Mau',
            'english cocker spaniel',
            'english setter',
            'german shorthaired',
            'great pyrenees',
            'havanese',
            'japanese chin',
            'keeshond',
            'leonberger',
            'Maine Coon',
            'miniature pinscher',
            'newfoundland',
            'Persian',
            'pomeranian',
            'pug',
            'Ragdoll',
            'Russian Blue',
            'saint bernard',
            'samoyed',
            'scottish terrier',
            'shiba inu',
            'Siamese',
            'Sphynx',
            'staffordshire bull terrier',
            'wheaten terrier',
            'yorkshire terrier',
        ]
        if base_set is not None:
            self.dataset = base_set
            self.targets = base_set['class_id'].unique()
        else:
            self._load_metadata()

    def __getitem__(self, idx):

        sample = self.dataset.iloc[idx]
        path = os.path.join(self.root, 'images', sample.img_id) + '.jpg'

        target = sample.class_id - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target#, idx
    
    def _load_metadata(self):
        if self.train:
            train_file = os.path.join(self.root, 'annotations', 'trainval.txt')
            self.dataset = pd.read_csv(train_file, sep=' ', names=['img_id', 'class_id', 'species', 'breed_id'])
        else:
            test_file = os.path.join(self.root, 'annotations', 'test.txt')
            self.dataset = pd.read_csv(test_file, sep=' ', names=['img_id', 'class_id', 'species', 'breed_id'])
        
        self.targets = self.dataset['class_id'].unique()

    def __len__(self):
        return len(self.dataset)

def prepare_train_loaders(config):
    return {
        'full': torch.utils.data.DataLoader(
            OxfordPets(LOCATION, train=True, transform=config['train_preprocess']), 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers']
        )
    }

def prepare_test_loaders(config):
    test_set = OxfordPets(LOCATION, train=False, transform=config['eval_preprocess'])
    loaders = {
        'test': torch.utils.data.DataLoader(
            test_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    }
    loaders['class_names'] = test_set.classnames
    
    from dataset_parsers.label_checks import verify_labels
    if not verify_labels('pets', loaders['class_names']):
        pdb.set_trace
    return loaders
