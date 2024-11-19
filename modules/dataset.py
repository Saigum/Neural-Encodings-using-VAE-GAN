import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from torch.utils.data import Subset

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
set_seed(42)

class CelebADataset(Dataset):
    def __init__(self, root_dir, partition_file, attr_file, transform=None):
        """
        root_dir: path containing img_align_celeba folder
        partition_file: path to list_eval_partition.csv
        attr_file: path to list_attr_celeba.csv
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img_align_celeba', 'img_align_celeba')  # Note the double img_align_celeba
        
        # Load partition info (0=train, 1=val, 2=test)
        self.partition_df = pd.read_csv(partition_file)
        
        # Load attributes
        self.attr_df = pd.read_csv(attr_file)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.partition_df)
    
    def __getitem__(self, idx):
        # Get image name
        img_name = os.path.join(self.img_dir, self.partition_df.iloc[idx, 0])  # Changed from attr_df to partition_df
        
        # Load image
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
            
        return image

def get_celeba_dataloaders(root_dir, batch_size=64, subset_fraction=0.1):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset
    full_dataset = CelebADataset(
        root_dir=root_dir,
        partition_file=os.path.join(root_dir, 'list_eval_partition.csv'),
        attr_file=os.path.join(root_dir, 'list_attr_celeba.csv'),
        transform=transform
    )
    
    # Get indices for train/val/test from partition file
    partition_df = pd.read_csv(os.path.join(root_dir, 'list_eval_partition.csv'))
    train_indices = partition_df[partition_df['partition'] == 0].index.tolist()
    val_indices = partition_df[partition_df['partition'] == 1].index.tolist()
    test_indices = partition_df[partition_df['partition'] == 2].index.tolist()
    
    # Take subset of train indices
    num_train_samples = int(len(train_indices) * subset_fraction)
    train_indices = train_indices[:num_train_samples]
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


