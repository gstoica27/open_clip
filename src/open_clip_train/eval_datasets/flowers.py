import torch
import torchvision
import pdb


# LOCATION = '/gscratch/krishna/gstoica3/datasets/flowers'
LOCATION = '/weka/prior-default/georges/research/MergedVisionEncoders/datasets/flowers'

classnames = [
    'pink primrose',
    'hard-leaved pocket orchid',
    'canterbury bells',
    'sweet pea',
    'english marigold',
    'tiger lily',
    'moon orchid',
    'bird of paradise',
    'monkshood',
    'globe thistle',
    'snapdragon',
    "colt's foot",
    'king protea',
    'spear thistle',
    'yellow iris',
    'globe flower',
    'purple coneflower',
    'peruvian lily',
    'balloon flower',
    'giant white arum lily',
    'fire lily',
    'pincushion flower',
    'fritillary',
    'red ginger',
    'grape hyacinth',
    'corn poppy',
    'prince of wales feathers',
    'stemless gentian',
    'artichoke',
    'sweet william',
    'carnation',
    'garden phlox',
    'love in the mist',
    'mexican aster',
    'alpine sea holly',
    'ruby-lipped cattleya',
    'cape flower',
    'great masterwort',
    'siam tulip',
    'lenten rose',
    'barbeton daisy',
    'daffodil',
    'sword lily',
    'poinsettia',
    'bolero deep blue',
    'wallflower',
    'marigold',
    'buttercup',
    'oxeye daisy',
    'common dandelion',
    'petunia',
    'wild pansy',
    'primula',
    'sunflower',
    'pelargonium',
    'bishop of llandaff',
    'gaura',
    'geranium',
    'orange dahlia',
    'pink and yellow dahlia',
    'cautleya spicata',
    'japanese anemone',
    'black-eyed susan',
    'silverbush',
    'californian poppy',
    'osteospermum',
    'spring crocus',
    'bearded iris',
    'windflower',
    'tree poppy',
    'gazania',
    'azalea',
    'water lily',
    'rose',
    'thorn apple',
    'morning glory',
    'passion flower',
    'lotus',
    'toad lily',
    'anthurium',
    'frangipani',
    'clematis',
    'hibiscus',
    'columbine',
    'desert-rose',
    'tree mallow',
    'magnolia',
    'cyclamen',
    'watercress',
    'canna lily',
    'hippeastrum',
    'bee balm',
    'air plant',
    'foxglove',
    'bougainvillea',
    'camellia',
    'mallow',
    'mexican petunia',
    'bromelia',
    'blanket flower',
    'trumpet creeper',
    'blackberry lily',
]

def prepare_train_loaders(config):
    dataset = torchvision.datasets.Flowers102(
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
    test_set = torchvision.datasets.Flowers102(
        root=LOCATION,
        split='val',
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
    loaders['class_names'] = classnames
    from open_clip_train.eval_datasets.label_checks import verify_labels
    if not verify_labels('flowers', loaders['class_names']):
        pdb.set_trace
    return loaders