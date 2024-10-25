from torch import nn
from torch.nn import functional as F
import torch as th  



class Encoder(nn.Module):
    def __init__(self,image_size,num_channels,latent_dim,hidden_dim):
        ''' Latent dim represents the dimension of the latent space.
        The final dimension will be 1x1xlatent_dim'''
        super(Encoder,self).__init__( )
        self.image_size = image_size
        ''' Image size will be 128*128*3'''
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encode_block = nn.Sequential(
            [nn.Conv2d(in_channels=num_channels,
                                out_channels=192,kernel_size=3,
                                stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=192,out_channels=256,
                                kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=256,out_channels=384,
                                kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=384,out_channels=512,
                                kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=512,out_channels=768,
                                kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=768,out_channels=1024,
                                kernel_size=3,stride=2,padding=1),
            nn.Flatten(),
            nn.Linear(in_features=1024,out_features=1024),]
        )
        self.mu = nn.Linear(in_features=1024,out_features=latent_dim)
        self.log_var = nn.Linear(in_features=1024,out_features=latent_dim)
    def forward(self,x):
        x = self.encode_block(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        self.kl_divergence = 0.5 * th.sum(th.exp(log_var) + mu**2 - 1 - log_var)
        return mu,log_var


                                              
                                              
                                              
                                            