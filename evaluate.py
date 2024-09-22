import torch
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            pred_masks = (outputs > 0.5).float()

            plt.subplot(1, 3, 1)
            plt.imshow(images[0].cpu().permute(1, 2, 0))
            plt.subplot(1, 3, 2)
            plt.imshow(masks[0].cpu().squeeze(), cmap='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(pred_masks[0].cpu().squeeze(), cmap='gray')
            plt.show()
