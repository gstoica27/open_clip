import os
import torch
import torchvision.datasets as datasets
import numpy as np
import pdb


# LOCATION = '/gscratch/krishna/gstoica3/datasets'
LOCATION = "/weka/prior-default/georges/research/MergedVisionEncoders/datasets"

class SUN397:
    def __init__(self,
                 is_train,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        traindir = os.path.join(location, 'SUN397', 'train')
        valdir = os.path.join(location, 'SUN397', 'val')


        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]

def prepare_train_loaders(config):
    dataset_class = SUN397(
        is_train=True,
        preprocess=config['train_preprocess'],
        location=LOCATION,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    loaders = {
        'full': dataset_class.train_loader
    }
    return loaders

def prepare_test_loaders(config):
    dataset_class = SUN397(
        is_train=False,
        preprocess=config['eval_preprocess'],
        location=LOCATION,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    loaders = {
        'test': dataset_class.test_loader
    }
    if config.get('val_fraction', 0) > 0.:
        print('splitting sun397')
        test_set = loaders['test'].dataset
        # test_set, val_set = create_heldout_split(test_set, config['val_fraction'])
        shuffled_idxs = np.random.permutation(np.arange(len(test_set)))
        num_valid = int(len(test_set) * config['val_fraction'])
        valid_idxs, test_idxs = shuffled_idxs[:num_valid], shuffled_idxs[num_valid:]
        
        # test_set, val_set = create_heldout_split(test_set, config['val_fraction'])
        val_set =  torch.utils.data.Subset(test_set, valid_idxs)
        test_set =  torch.utils.data.Subset(test_set, test_idxs)
        loaders['test'] = torch.utils.data.DataLoader(
            test_set,
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        loaders['val'] = torch.utils.data.DataLoader(
            val_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    loaders['class_names'] = dataset_class.classnames
    from dataset_parsers.label_checks import verify_labels
    if not verify_labels('sun397', loaders['class_names']):
        pdb.set_trace
        
    return loaders
