# Import necessary modules
import torch
import os
from PIL import Image

# RTTS dataset class
class RTTSDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, image_dir, transform=None):
        self.image_list = image_list
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image