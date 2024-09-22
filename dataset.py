import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, device='cpu'):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))  # Change to channel first
        image = image / 255.0  # Normalize

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        mask = mask / 255.0  # Normalize to [0,1]

        image = torch.tensor(image, dtype=torch.float32).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).to(self.device)

        return image, mask
