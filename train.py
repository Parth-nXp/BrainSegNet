import torch
import torch.optim as optim

def dice_coefficient(y_true, y_pred, smooth=100):
    y_true_flatten = y_true.view(-1)
    y_pred_flatten = y_pred.view(-1)
    intersection = (y_true_flatten * y_pred_flatten).sum()
    union = y_true_flatten.sum() + y_pred_flatten.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def dice_coefficient_loss(y_true, y_pred, smooth=100):
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def iou(y_true, y_pred, smooth=100):
    intersection = (y_true * y_pred).sum()
    union = (y_true + y_pred).sum()
    return (intersection + smooth) / (union - intersection + smooth)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, scheduler):
    train_loss_list = []
    val_loss_list = []
    train_dice_list = []
    val_dice_list = []
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            dice = dice_coefficient(outputs, masks).item()
            running_loss += loss.item()
            running_dice += dice
            
        avg_train_loss = running_loss / len(train_loader)
        avg_train_dice = running_dice / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                dice = dice_coefficient(outputs, masks
