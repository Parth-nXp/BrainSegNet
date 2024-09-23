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
            pred
