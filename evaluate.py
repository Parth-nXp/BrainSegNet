# evaluate.py

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

def evaluate_model(model, df_test, device, im_height, im_width):
    model.eval()
    
    for i in range(5):
        index = np.random.randint(0, len(df_test))
        img_path = df_test.iloc[index]['image_filename']
        mask_path = df_test.iloc[index]['mask_images']
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (im_height, im_width))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_tensor).cpu().squeeze().numpy() > 0.5
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (im_height, im_width))
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Original Mask')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='gray')
        plt.title('Prediction')

        plt.show()


       
