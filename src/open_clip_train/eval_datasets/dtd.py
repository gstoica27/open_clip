import os
import torch
import torchvision.datasets as datasets
import pdb


# LOCATION = "/gscratch/krishna/gstoica3/datasets/dtd"
LOCATION = "/weka/prior-default/georges/research/MergedVisionEncoders/datasets/dtd"

class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('/gscratch/krishna/gstoica3/datasets'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        traindir = os.path.join(location, 'dtd', 'train')
        valdir = os.path.join(location, 'dtd', 'val')

        self.train_dataset = datasets.ImageFolder(
            traindir, transform=preprocess)
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
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]
        

def prepare_train_loaders(config):
    train_dataset = datasets.ImageFolder(
        os.path.join(LOCATION, "train"), 
        transform=config['train_preprocess']
    )
    
    loaders = {
        'full': torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            num_workers=config['num_workers']
        )
    }
    
    return loaders

def prepare_test_loaders(config):
    
    val_dataset = datasets.ImageFolder(
        os.path.join(LOCATION, "val"), 
        transform=config['eval_preprocess']
    )
    # test_dataset = datasets.ImageFolder(
    #     os.path.join(LOCATION, "test"), 
    #     transform=config['eval_preprocess']
    # )
    
    idx_to_class = dict((v, k)
                            for k, v in val_dataset.class_to_idx.items())
    classnames = [idx_to_class[i].replace(
        '_', ' ') for i in range(len(idx_to_class))]
    
    loaders = {
        'test': torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        ),
        # 'test': torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_size=config['batch_size'],
        #     num_workers=config['num_workers']
        # ),
        'class_names': classnames
    }
    from open_clip_train.eval_datasets.label_checks import verify_labels
    if not verify_labels('dtd', loaders['class_names']):
        pdb.set_trace
    
    return loaders