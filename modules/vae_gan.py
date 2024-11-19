import torch
from torch import nn
from torch.nn import functional as F
from modules.encoder import Encoder
from modules.decoder import Decoder

class VAE_GAN(nn.Module):
   def __init__(self, image_size=128, num_channels=3, latent_dim=1024, hidden_dim=1024):
       super(VAE_GAN, self).__init__()
       
       self.encoder = Encoder(image_size, num_channels, latent_dim, hidden_dim)
       self.decoder = Decoder(image_size, num_channels, latent_dim, hidden_dim)
       
       # Discriminator network
       self.discriminator = nn.Sequential(
           # input: [batch, 3, 128, 128] -> output: [batch, 64, 64, 64]
           nn.Conv2d(num_channels, 64, kernel_size=4, stride=2, padding=1),
           nn.ELU(),
           
           # output: [batch, 64, 32, 32]
           nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
           nn.ELU(),
           
           # output: [batch, 64, 16, 16]
           nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
           nn.ELU(),
           
           # output: [batch, 64, 8, 8]
           nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
           nn.ELU(),
           
           # output: [batch, 64, 4, 4]
           nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
           nn.ELU(),
           
           # output: [batch, 1]
           nn.Flatten(),
           nn.Linear(64 * 4 * 4, 1),
           nn.Sigmoid()
       )
       
   def reparameterize(self, mu, log_var):
       """
       Sample z from the given mu and log_var
       """
       std = torch.exp(0.5 * log_var)
       eps = torch.randn_like(std)
       return mu + eps * std
       
   def forward(self, x, compute_loss=False):
       # generate mu, var (encoder)
       mu, log_var = self.encoder(x)
       
       # Sample from latent space
       z = self.reparameterize(mu, log_var)
       
       # reconstruction
       x_recon = self.decoder(z)
       
       # Discriminate both real and reconstructed images
       d_real = self.discriminator(x)
       d_fake = self.discriminator(x_recon)
       
       if compute_loss:
           #  maybe useful later
           # Reconstruction loss (feature matching in discriminator space)
           recon_loss = F.mse_loss(d_fake, d_real)
           
           # KL divergence
           kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
           
           # Adversarial losses
           d_loss = -torch.mean(torch.log(d_real) + torch.log(1 - d_fake))
           g_loss = -torch.mean(torch.log(d_fake))
           
           return x_recon, {
               'reconstruction_loss': recon_loss,
               'kl_loss': kl_loss,
               'discriminator_loss': d_loss,
               'generator_loss': g_loss
           }
           
       return x_recon

   def encode(self, x):
       mu, log_var = self.encoder(x)
       return mu, log_var
       
   def decode(self, z):
       return self.decoder(z)
       
   def generate(self, num_samples):
       """
       Generate samples from random latent vectors
       """
       z = torch.randn(num_samples, self.encoder.latent_dim)
       return self.decode(z)