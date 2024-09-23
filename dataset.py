import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class BrainMRIDataset(Dataset):
    def __init__(self, dataframe, image_transform=None, mask_transform=None, target_size=(256, 256)):
        self.dataframe = dataframe
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['image_filename']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = self.dataframe.iloc[idx]['mask_images']
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)
        
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        image = Image.fromarray((image * 255).astype(np.uint8))
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        
        if self.image_transform:
            image = self.image_transform(image)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        mask = mask.unsqueeze(0)
        
        return image, mask
