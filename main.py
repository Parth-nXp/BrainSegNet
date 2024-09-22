# main.py

import torch
from torch.utils.data import DataLoader
from dataset import BrainMRIDataset
from model import UNet
from train import train_model
from evaluate import evaluate_model
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

def main():
    # Prepare data
    mask_paths = glob('./Dataset/kaggle_3m/*/*_mask*')
    image_paths = [i.replace('_mask', '') for i in mask_paths]

    df = pd.DataFrame({'image_paths': image_paths, 'mask_paths': mask_paths})
    df_train, df_test = train_test_split(df, test_size=0.05)
    df_train, df_val = train_test_split(df_train, test_size=0.05)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets and dataloaders
    train_dataset = BrainMRIDataset(df_train['image_paths'].tolist(), df_train['mask_paths'].tolist(), device=device)
    val_dataset = BrainMRIDataset(df_val['image_paths'].tolist(), df_val['mask_paths'].tolist(), device=device)
    test_dataset = BrainMRIDataset(df_test['image_paths'].tolist(), df_test['mask_paths'].tolist(), device=device)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    # Model initialization
    model = UNet().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Train the model
    train_model(model, train_loader, optimizer, criterion)

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')

    # Load the saved model
    model.load_state_dict(torch.load('model.pth'))

    # Evaluate the model
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
