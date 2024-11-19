import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd


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


def show_batch(dataloader, n=16, nrow=4, num_attributes=5):
   """
   Show n images from dataloader in a grid with their top attributes
   num_attributes: number of most common attributes to show for each image
   """
   import matplotlib.pyplot as plt
   
   # Get a batch
   images = next(iter(dataloader))
   
   # Get attribute names from the dataset
   attr_file = '/kaggle/input/celeba-dataset/list_attr_celeba.csv'
   attr_df = pd.read_csv(attr_file)
   attr_names = attr_df.columns[1:].tolist()  # Skip image_id column
   
   # Convert from [-1,1] to [0,1] for display
   images = (images + 1) / 2
   
   # Create a grid
   fig, axes = plt.subplots(nrow, nrow, figsize=(15, 15))
   for i, ax in enumerate(axes.flat):
       if i < n:
           # Display image
           img = images[i].permute(1,2,0).cpu().numpy()
           ax.imshow(img)
           
           # Get the corresponding attributes
           idx = dataloader.dataset.indices[i]  # Get original index
           attrs = attr_df.iloc[idx, 1:].astype(int)  # Get attributes (skip image_id)
           
           # Get the top attributes (where value is 1)
           pos_attrs = [name for name, val in zip(attr_names, attrs) if val == 1]
           attrs_text = '\n'.join(pos_attrs[:num_attributes])  # Show top N attributes
           
           # Add attributes as title
           ax.set_title(attrs_text, fontsize=8, pad=5)
           ax.axis('off')
   
   plt.tight_layout()
   plt.show()
   
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']