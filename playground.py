from encoder import Encoder
from PIL import Image
from utils import preprocess_image
IMAGE_PATH = "example.jpg"

encoder = Encoder(image_size=128, num_channels=3, latent_dim=1024, hidden_dim=1024)



try:
    input_tensor = preprocess_image(IMAGE_PATH)
    
    z, mu, log_var = encoder(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Z shape:", z.shape)
    print("Mu shape:", mu.shape)
    print("Log_var shape:", log_var.shape)
    print("KL divergence:", encoder.kl_divergence.item())

except Exception as e:
    print(f"Error: {str(e)}")