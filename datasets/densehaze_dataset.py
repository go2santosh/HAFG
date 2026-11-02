# Import necessary modules
import torch
import os
import random
from PIL import Image

# Dense Haze dataset class
class DenseHazeDataset(torch.utils.data.Dataset):
    def __init__(self, hazy_images, hazy_dir, clear_dir, transform=None, return_image_name=False, augment=False, patch_size=None):
        self.hazy_images = hazy_images
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.return_image_name = return_image_name
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        clear_path = os.path.join(self.clear_dir, self.hazy_images[idx].split('_')[0] + '_GT.png')
        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        # --- Data augmentation ---
        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                hazy_img = hazy_img.transpose(Image.FLIP_LEFT_RIGHT)
                clear_img = clear_img.transpose(Image.FLIP_LEFT_RIGHT)

            # Random cropping (same region)
            if self.patch_size:
                w, h = hazy_img.size
                if w > self.patch_size and h > self.patch_size:
                    left = random.randint(0, w - self.patch_size)
                    top = random.randint(0, h - self.patch_size)
                    box = (left, top, left + self.patch_size, top + self.patch_size)
                    hazy_img = hazy_img.crop(box)
                    clear_img = clear_img.crop(box)

        # --- Apply transforms (ToTensor, Normalize, etc.) ---
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clear_img = self.transform(clear_img)
        else:
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            hazy_img = to_tensor(hazy_img)
            clear_img = to_tensor(clear_img)
        
        if self.return_image_name:
            return hazy_img, clear_img, self.image_list[idx].split('_')[0]
        else:
            return hazy_img, clear_img