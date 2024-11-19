from torch import nn


class Decoder(nn.Module):
   def __init__(self, image_size=128, num_channels=3, latent_dim=1024, hidden_dim=1024):
       ''' From latent dim to image size reconstruction '''
       super(Decoder, self).__init__()
       
       self.image_size = image_size
       ''' Image size will be 128*128*3 '''
       self.num_channels = num_channels
       self.latent_dim = latent_dim
       self.hidden_dim = hidden_dim
       
       self.decode_block = nn.Sequential(
           # input: [batch, 1024] -> output: [batch, 1024, 2, 2]
           nn.Linear(latent_dim, 1024 * 2 * 2),
           nn.ELU(),
           nn.Unflatten(1, (1024, 2, 2)),
           
           # output: [batch, 768, 4, 4]
           nn.ConvTranspose2d(1024, 768, kernel_size=3, stride=2, padding=1, output_padding=1),
           nn.BatchNorm2d(768),
           nn.ELU(),
           
           # output: [batch, 512, 8, 8]
           nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
           nn.BatchNorm2d(512),
           nn.ELU(),
           
           # output: [batch, 384, 16, 16]
           nn.ConvTranspose2d(512, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
           nn.BatchNorm2d(384),
           nn.ELU(),
           
           # output: [batch, 256, 32, 32]
           nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
           nn.BatchNorm2d(256),
           nn.ELU(),
           
           # output: [batch, 192, 64, 64]
           nn.ConvTranspose2d(256, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
           nn.BatchNorm2d(192),
           nn.ELU(),
           
           # output: [batch, 3, 128, 128]
           nn.ConvTranspose2d(192, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
           nn.Tanh()  # or Sigmoid() depending on your image scaling
       )
       
   def forward(self, z):
       return self.decode_block(z)
