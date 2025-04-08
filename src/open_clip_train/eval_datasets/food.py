# /gscratch/krishna/gstoica3/datasets/food/food-101
import torch
import torchvision
import pdb
import pdb


# LOCATION = "/gscratch/krishna/gstoica3/datasets/food"
LOCATION = "/weka/prior-default/georges/research/MergedVisionEncoders/datasets/food"


def prepare_train_loaders(config):
    dataset = torchvision.datasets.Food101(
        root=LOCATION,
        split='train',
        transform=config['train_preprocess'],
        download=True
    )
    return {
        'full': torch.utils.data.DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    }

def prepare_test_loaders(config):
    test_set = torchvision.datasets.Food101(
        root=LOCATION,
        split='test',
        transform=config['eval_preprocess'],
        download=True
    )
    loaders = {
        'test': torch.utils.data.DataLoader(
            test_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    }
    loaders['class_names'] = [' '.join(i.split('_')) for i in test_set.classes]
    from open_clip_train.eval_datasets.label_checks import verify_labels
    if not verify_labels('food', loaders['class_names']):
        pdb.set_trace
    # pdb.set_trace()
    return loaders
