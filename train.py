import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from modules.dataset import get_celeba_dataloaders
from modules.vae_gan import VAE_GAN


root_dir = '/kaggle/input/celeba-dataset' # PATH TO CELEBA DATASET (run on kaggle)
train_loader, val_loader, test_loader = get_celeba_dataloaders(
    root_dir=root_dir,
    batch_size=64,
    subset_fraction=1.0
)



def train_vaegan(model, train_loader, num_epochs=15, device="cuda", 
                lr=0.0001, beta1=0.5, beta2=0.999, checkpoint_dir='checkpoints'):
    """
    Train the VAE-GAN model following Larsen et al. 2016
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'samples'), exist_ok=True)
    
    model = model.to(device)
    
    # Create optimizers for each component
    opt_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lr, betas=(beta1, beta2))
    opt_decoder = torch.optim.Adam(model.decoder.parameters(), lr=lr, betas=(beta1, beta2))
    opt_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Add schedulers with increased patience as per paper suggestions
    scheduler_e = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_encoder, mode='min', factor=0.5, patience=5)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_decoder, mode='min', factor=0.5, patience=5)
    scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_discriminator, mode='min', factor=0.5, patience=5)
    
    lambda_kl = 0.01
    lambda_rec = 1.0
    # lambda_adv = 0.1
    
    # Keep track of best loss for checkpoint saving
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_d_loss = 0
        total_g_loss = 0
        total_vae_loss = 0
        
        model.train()
        for batch_idx, real_images in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels for adversarial training with label smoothing
            real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # Label smoothing
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            #========================= Train Discriminator =========================
            opt_discriminator.zero_grad()
            
            mu, log_var = model.encoder(real_images)
            z = model.reparameterize(mu, log_var)
            fake_images = model.decoder(z)
            
            d_real = model.discriminator(real_images)
            d_fake = model.discriminator(fake_images.detach())
            
            # Discriminator loss (not weighted by lambda_adv as per paper)
            d_loss_real = F.binary_cross_entropy(d_real, real_labels)
            d_loss_fake = F.binary_cross_entropy(d_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            
            d_loss.backward()
            opt_discriminator.step()
            
            #====================== Train Generator/Decoder ======================
            opt_decoder.zero_grad()
            
            d_fake = model.discriminator(fake_images)
            g_loss = F.binary_cross_entropy(d_fake, real_labels)  # Not weighted by lambda_adv
            
            g_loss.backward()
            opt_decoder.step()
            
            #====================== Train VAE (Encoder-Decoder) ======================
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            
            mu, log_var = model.encoder(real_images)
            z = model.reparameterize(mu, log_var)
            fake_images = model.decoder(z)
            
            # Get discriminator features for reconstruction loss
            d_fake_features = model.discriminator(fake_images)
            d_real_features = model.discriminator(real_images)
            
            # Reconstruction loss uses discriminator features as per paper
            recon_loss = lambda_rec * F.mse_loss(d_fake_features, d_real_features)
            # KL loss normalized by batch size as per paper
            kl_loss = lambda_kl * (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / batch_size
            
            vae_loss = recon_loss + kl_loss
            
            vae_loss.backward()
            opt_encoder.step()
            opt_decoder.step()
            
            # Save losses for reporting
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            total_vae_loss += vae_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} '
                      f'VAE_loss: {vae_loss.item():.4f}')
        
        # Calculate average losses for epoch
        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)
        avg_vae_loss = total_vae_loss / len(train_loader)
        total_loss = avg_d_loss + avg_g_loss + avg_vae_loss
        
        print(f'Epoch [{epoch}/{num_epochs}] Average Losses: '
              f'D: {avg_d_loss:.4f} G: {avg_g_loss:.4f} VAE: {avg_vae_loss:.4f}')
        
        with torch.no_grad():
            # random samples  from latent space
            z = torch.randn(16, model.encoder.latent_dim).to(device)
            sample_images = model.decoder(z)
            save_image(sample_images.data[:16], 
                      os.path.join(checkpoint_dir, 'samples', f'sample_epoch_{epoch}.png'),
                      nrow=4, normalize=True)
                      
            # save reconstructions
            recons = model.decoder(model.encoder(real_images[:8])[0])
            comparison = torch.cat([real_images[:8], recons])
            save_image(comparison,
                      os.path.join(checkpoint_dir, 'samples', f'recon_epoch_{epoch}.png'),
                      nrow=8, normalize=True)
        
        # Step schedulers
        scheduler_e.step(avg_vae_loss)
        scheduler_d.step(avg_g_loss)
        scheduler_disc.step(avg_d_loss)
        
        # Save checkpoint if total loss improved
        if total_loss < best_loss:
            best_loss = total_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_optimizer': opt_encoder.state_dict(),
                'decoder_optimizer': opt_decoder.state_dict(),
                'discriminator_optimizer': opt_discriminator.state_dict(),
                'loss': total_loss,
                'scheduler_states': {
                    'encoder': scheduler_e.state_dict(),
                    'decoder': scheduler_d.state_dict(),
                    'discriminator': scheduler_disc.state_dict()
                }
            }
            torch.save(checkpoint, f'{checkpoint_dir}/best_model.pt')
        
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_optimizer': opt_encoder.state_dict(),
                'decoder_optimizer': opt_decoder.state_dict(),
                'discriminator_optimizer': opt_discriminator.state_dict(),
                'loss': total_loss,
                'scheduler_states': {
                    'encoder': scheduler_e.state_dict(),
                    'decoder': scheduler_d.state_dict(),
                    'discriminator': scheduler_disc.state_dict()
                }
            }
            torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt')
            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE_GAN().to(device)
train_vaegan(
    model=model,
    train_loader=train_loader,
    num_epochs=15,
    device=device,
    checkpoint_dir='checkpoints'
)