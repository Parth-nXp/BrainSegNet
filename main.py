import torch
from dataset import BrainMRIDataset
from model import UNet
from train import train_model, dice_coefficient_loss
from evaluate import evaluate_model
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from glob import glob
from torchvision import transforms

# Set parameters
im_height, im_width = 256, 256
batch_size = 4
epochs = 150
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
image_filenames = [i.replace("_mask", "") for i in glob("./MRI_Dataset/*/*_mask*")]
mask_images = glob("./MRI_Dataset/*/*_mask*")
df = pd.DataFrame({'image_filename': image_filenames, 'mask_images': mask_images})

df_train, df_test = train_test_split(df, test_size=0.1)

# Define transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((im_height, im_width)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([transforms.ToTensor()])

# Create datasets and loaders
train_dataset = BrainMRIDataset(df_train, train_transforms, mask_transform)
val_dataset = BrainMRIDataset(df_test, train_transforms, mask_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and scheduler
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Train model
train_loss, val_loss, train_dice, val_dice = train_model(
    model, train_loader, val_loader, dice_coefficient_loss, optimizer, epochs, device, scheduler
)

# Evaluate model
evaluate_model(model, val_loader, device, im_height, im_width)
