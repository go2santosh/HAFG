# Import necessary modules
import torch
import os
from PIL import Image

# OTS dataset class
class OTSDataset(torch.utils.data.Dataset):
    def __init__(self, hazy_images, hazy_dir, clear_dir, transform=None):
        self.hazy_images = hazy_images
        # Create pairs of hazy names and index with file existence check filter for clear images not existing
        self.hazy_clean_pairs = [(i, hazy_name, f"{hazy_name.split('_')[0]}.jpg") for i, hazy_name in enumerate(hazy_images) if os.path.isfile(os.path.join(clear_dir, f"{hazy_name.split('_')[0]}.jpg"))]
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform

    def __len__(self):
        return len(self.hazy_clean_pairs)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_clean_pairs[idx][1])
        clear_path = os.path.join(self.clear_dir, self.hazy_clean_pairs[idx][2])
        hazy = Image.open(hazy_path).convert('RGB')
        clear = Image.open(clear_path).convert('RGB')
        if self.transform:
            hazy = self.transform(hazy)
            clear = self.transform(clear)
        # Ensure the dataset returns exactly two values
        return hazy, clear