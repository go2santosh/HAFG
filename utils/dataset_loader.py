from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os
import sys
import random
sys.path.insert(0, "..")

from datasets.its_dataset import ITSDataset
from datasets.ots_dataset import OTSDataset
from datasets.sots_dataset import SOTSDataset
from datasets.nhhaze_dataset import NHNazeDataset
from datasets.densehaze_dataset import DenseHazeDataset
from datasets.rtts_dataset import RTTSDataset

def get_dataset_loader(config, dataset_name, source_images=None):
    # Set dataset paths
    if dataset_name == "rtts":
        config.update({
            'hazy_train_dir': "../../../datasets/RTTS/JPEGImages"
        })
    elif dataset_name == "sots_indoor":
        config.update({
            'hazy_train_dir': "../../../datasets/SOTS/indoor/hazy",
            'clear_train_dir': "../../../datasets/SOTS/indoor/clear"
        })
    elif dataset_name == "sots_outdoor":
        config.update({
            'hazy_train_dir': "../../../datasets/SOTS/outdoor/hazy",
            'clear_train_dir': "../../../datasets/SOTS/outdoor/clear"
        })
    elif dataset_name == "nhhaze":
        config.update({
            'hazy_train_dir': "../../../datasets/NH-HAZE/train/hazy",
            'clear_train_dir': "../../../datasets/NH-HAZE/train/clear",
            'hazy_eval_dir': "../../../datasets/NH-HAZE/val/hazy",
            'clear_eval_dir': "../../../datasets/NH-HAZE/val/clear",
            'hazy_test_dir': "../../../datasets/NH-HAZE/test/hazy",
            'clear_test_dir': "../../../datasets/NH-HAZE/test/clear"
        })
    elif dataset_name == "densehaze":
        config.update({
            'hazy_train_dir': "../../../datasets/Dense-Haze/train/hazy",
            'clear_train_dir': "../../../datasets/Dense-Haze/train/GT",
            'hazy_eval_dir': "../../../datasets/Dense-Haze/val/hazy",
            'clear_eval_dir': "../../../datasets/Dense-Haze/val/GT",
            'hazy_test_dir': "../../../datasets/Dense-Haze/test/hazy",
            'clear_test_dir': "../../../datasets/Dense-Haze/test/GT"
        })
    elif dataset_name == "its":
        config.update({
            'hazy_train_dir': "../../../datasets/ITS/hazy",
            'clear_train_dir': "../../../datasets/ITS/clear",
            'hazy_test_dir': "../../../datasets/SOTS/indoor/hazy",
            'clear_test_dir': "../../../datasets/SOTS/indoor/clear"
        })
    elif dataset_name == "ots":
        config.update({
            'hazy_train_dir': "../../../datasets/OTS/hazy",
            'clear_train_dir': "../../../datasets/OTS/clear",
            'hazy_test_dir': "../../../datasets/SOTS/outdoor/hazy",
            'clear_test_dir': "../../../datasets/SOTS/outdoor/clear"
        })
    # Transforms (resize filter: bilinear or bicubic)
    interpolation = {'bilinear': transforms.InterpolationMode.BILINEAR,
                     'bicubic': transforms.InterpolationMode.BICUBIC}[config.get('resize', 'bicubic')]
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Creating a new transform for testing
    transformTest = transforms.Compose([
        transforms.Resize((config['test_image_size'], config['test_image_size']), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Data loaders
    hazy_train_images = source_images if source_images is not None else os.listdir(config['hazy_train_dir'])
    # Optional random subset of the training images (e.g. 1000 OTS pairs), fixed by the seed
    if config.get('train_subset', 0) and dataset_name != "rtts" and len(hazy_train_images) > config['train_subset']:
        hazy_train_images = random.Random(config.get('seed', 42)).sample(sorted(hazy_train_images), config['train_subset'])

    if dataset_name == "rtts":
        # Data loaders for RTTS dataset (real hazy images without ground truth: test only)
        train_dataset = None
        train_loader = None
        eval_dataset = None
        eval_loader = None
        test_dataset = RTTSDataset(hazy_train_images, config['hazy_train_dir'], transform=transformTest)
        test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)
    elif dataset_name in ["sots_indoor_testonly", "sots_outdoor_testonly"]:
        # Data loaders for SOTS dataset
        train_dataset = None
        train_loader = None
        eval_dataset = None
        eval_loader = None
        test_dataset = SOTSDataset(hazy_train_images, config['hazy_train_dir'], config['clear_train_dir'], transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)
    elif dataset_name in ["sots_indoor", "sots_outdoor"]:
        # Data loaders for SOTS dataset
        train_images, eval_and_test_images = train_test_split(hazy_train_images, test_size=0.2, random_state=42)  # Fixed random_state
        eval_images, test_images = train_test_split(eval_and_test_images, test_size=0.5, random_state=42)  # Fixed random_state
        train_dataset = SOTSDataset(train_images, config['hazy_train_dir'], config['clear_train_dir'], transform=transform, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
        eval_dataset = SOTSDataset(eval_images, config['hazy_train_dir'], config['clear_train_dir'], transform=transform)
        eval_loader = DataLoader(eval_dataset, batch_size=config['test_batch_size'], shuffle=True)
        test_dataset = SOTSDataset(test_images, config['hazy_train_dir'], config['clear_train_dir'], transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)
    elif dataset_name == "its":
        # Data loaders for ITS dataset
        train_images, eval_images = train_test_split(hazy_train_images, test_size=0.2, random_state=42)  # Fixed random_state
        train_dataset = ITSDataset(train_images, config['hazy_train_dir'], config['clear_train_dir'], transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
        eval_dataset = ITSDataset(eval_images, config['hazy_train_dir'], config['clear_train_dir'], transform=transform)
        eval_loader = DataLoader(eval_dataset, batch_size=config['test_batch_size'], shuffle=True)
        hazy_test_images = os.listdir(config['hazy_test_dir'])
        test_dataset = SOTSDataset(hazy_test_images, config['hazy_test_dir'], config['clear_test_dir'], transform=transformTest)
        test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)
    elif dataset_name == "ots":
        # Data loaders for OTS dataset: trained on OTS, evaluated after each epoch and tested on SOTS outdoor
        train_dataset = OTSDataset(hazy_train_images, config['hazy_train_dir'], config['clear_train_dir'], transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
        hazy_test_images = os.listdir(config['hazy_test_dir'])
        eval_dataset = SOTSDataset(hazy_test_images, config['hazy_test_dir'], config['clear_test_dir'], transform=transform)
        eval_loader = DataLoader(eval_dataset, batch_size=config['test_batch_size'], shuffle=False)
        test_dataset = SOTSDataset(hazy_test_images, config['hazy_test_dir'], config['clear_test_dir'], transform=transformTest)
        test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)
    elif dataset_name == "nhhaze":
        # Data loaders for NH-Haze dataset
        train_dataset = NHNazeDataset(hazy_train_images, config['hazy_train_dir'], config['clear_train_dir'], transform=transform, augment=True, patch_size=256)
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=False)
        hazy_eval_images = os.listdir(config['hazy_eval_dir'])
        eval_dataset = NHNazeDataset(hazy_eval_images, config['hazy_eval_dir'], config['clear_eval_dir'], transform=transform)
        eval_loader = DataLoader(eval_dataset, batch_size=config['test_batch_size'], shuffle=False)
        hazy_test_images = os.listdir(config['hazy_test_dir'])
        test_dataset = NHNazeDataset(hazy_test_images, config['hazy_test_dir'], config['clear_test_dir'], transform=transformTest)
        test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)
    elif dataset_name == "densehaze":
        # Data loaders for NH-Haze or Dense-Haze dataset
        train_dataset = DenseHazeDataset(hazy_train_images, config['hazy_train_dir'], config['clear_train_dir'], transform=transform, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=False)
        hazy_eval_images = os.listdir(config['hazy_eval_dir'])
        eval_dataset = DenseHazeDataset(hazy_eval_images, config['hazy_eval_dir'], config['clear_eval_dir'], transform=transform)
        eval_loader = DataLoader(eval_dataset, batch_size=config['test_batch_size'], shuffle=False)
        hazy_test_images = os.listdir(config['hazy_test_dir'])
        test_dataset = DenseHazeDataset(hazy_test_images, config['hazy_test_dir'], config['clear_test_dir'], transform=transformTest)
        test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)
    
    return train_dataset, train_loader, eval_dataset, eval_loader, test_dataset, test_loader