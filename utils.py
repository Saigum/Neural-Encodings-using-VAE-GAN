from PIL import Image
import torchvision.transforms as transforms
import numpy as np



def preprocess_image(image_path):
    image = Image.open(image_path)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 128 x 128 input
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                           std=[0.5, 0.5, 0.5])  # normalize to [-1, 1]
    ])
    
    image_tensor = transform(image)
    
    image_tensor = image_tensor.unsqueeze(0)  # batch dimension
    
    return image_tensor