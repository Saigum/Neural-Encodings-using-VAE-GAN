from torch import nn
import torch


class Encoder(nn.Module):
   def __init__(self,image_size=128,num_channels=3,latent_dim=1024,hidden_dim=1024):
       ''' Latent dim represents the dimension of the latent space.
       The final dimension will be 1x1xlatent_dim'''
       super(Encoder,self).__init__()
       
       self.image_size = image_size
       ''' Image size will be 128*128*3'''
       self.num_channels = num_channels
       self.latent_dim = latent_dim
       self.hidden_dim = hidden_dim
       
       self.encode_block = nn.Sequential(
           # input: [batch, 3, 128, 128] -> output: [batch, 192, 64, 64]
           nn.Conv2d(num_channels, 192, kernel_size=3, stride=2, padding=1),
           nn.BatchNorm2d(192),
           nn.ELU(),
           
           # output: [batch, 256, 32, 32] 
           nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
           nn.BatchNorm2d(256),
           nn.ELU(),
           
           # output: [batch, 384, 16, 16]
           nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1), 
           nn.BatchNorm2d(384),
           nn.ELU(),
           
           # output: [batch, 512, 8, 8]
           nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1),
           nn.BatchNorm2d(512),
           nn.ELU(),
           
           # output: [batch, 768, 4, 4]
           nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=1),
           nn.BatchNorm2d(768),
           nn.ELU(),
           
           # output: [batch, 1024, 2, 2]
           nn.Conv2d(768, 1024, kernel_size=3, stride=2, padding=1),
           nn.BatchNorm2d(1024),
           nn.ELU(),
           
           # output: [batch, 1024]
           nn.Flatten(),
           nn.Linear(1024 * 2 * 2, hidden_dim),
           nn.ELU()
       )
       
       self.mu = nn.Linear(hidden_dim, latent_dim)
       self.log_var = nn.Linear(hidden_dim, latent_dim)
       
   def forward(self, x):
      x = self.encode_block(x)
      mu = self.mu(x)
      log_var = self.log_var(x)
      self.kl_divergence = 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1 - log_var)
      return mu,log_var


