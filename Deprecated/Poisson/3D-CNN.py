# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import numpy as np
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. UPGRADED HIGH-CAPACITY MODEL DEFINITION
# # ==========================================
# class PoissonAutoencoder3D(nn.Module):
#     def __init__(self, latent_dim=12):
#         super(PoissonAutoencoder3D, self).__init__()
        
#         # Encoder: 32x32x32 -> Latent Dim
#         self.encoder = nn.Sequential(
#             nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(32),
#             nn.LeakyReLU(0.2),
            
#             nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(64),
#             nn.LeakyReLU(0.2),
            
#             nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2),
            
#             nn.Flatten(),
#             nn.Linear(128 * 4 * 4 * 4, latent_dim)
#         )
        
#         # Decoder: Latent Dim -> 32x32x32
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(latent_dim, 128 * 4 * 4 * 4),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.decoder_conv = nn.Sequential(
#             nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(64),
#             nn.LeakyReLU(0.2),
            
#             nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(32),
#             nn.LeakyReLU(0.2),
            
#             nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, z):
#         x = self.decoder_fc(z)
#         x = x.view(-1, 128, 4, 4, 4) 
#         return self.decoder_conv(x)

#     def forward(self, x):
#         z = self.encode(x)
#         return self.decode(z)

# # ==========================================
# # 2. DATA GENERATION & NORMALIZATION
# # ==========================================
# def generate_poisson_dataset(num_samples, grid_size=32):
#     print(f"Generating {num_samples} samples for a {grid_size}^3 grid...")
    
#     x = np.linspace(0, 1, grid_size)
#     y = np.linspace(0, 1, grid_size)
#     z = np.linspace(0, 1, grid_size)
#     X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
#     U_data = np.zeros((num_samples, 1, grid_size, grid_size, grid_size))
#     F_data = np.zeros((num_samples, 1, grid_size, grid_size, grid_size))
#     scales = np.zeros(num_samples)
    
#     for i in range(num_samples):
#         kx = np.random.uniform(1.0, 10.0)
#         ky = np.random.uniform(1.0, 10.0)
#         kz = np.random.uniform(1.0, 10.0)
        
#         F = np.sin(2 * np.pi * kx * X) * np.sin(2 * np.pi * ky * Y) * np.sin(2 * np.pi * kz * Z)
#         C = 4 * np.pi**2 * (kx**2 + ky**2 + kz**2)
#         U = F / C
        
#         u_max = np.max(np.abs(U))
#         U_data[i, 0] = U / u_max
#         F_data[i, 0] = F
#         scales[i] = u_max
        
#     U_tensor = torch.tensor(U_data, dtype=torch.float32)
#     F_tensor = torch.tensor(F_data, dtype=torch.float32)
#     scales_tensor = torch.tensor(scales, dtype=torch.float32)
    
#     print(f"Per-sample normalization applied. Scale range: [{scales.min():.6f}, {scales.max():.6f}]")
    
#     return TensorDataset(U_tensor, F_tensor, scales_tensor), scales_tensor

# # ==========================================
# # 3. SETUP & TRAINING LOOP (WITH SCHEDULER)
# # ==========================================
# def main():
#     num_samples = 1000
#     batch_size = 32
#     epochs = 400  # Increased for higher accuracy
#     learning_rate = 1e-3
#     latent_dim = 12

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     dataset, scales = generate_poisson_dataset(num_samples=num_samples, grid_size=32)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     model = PoissonAutoencoder3D(latent_dim=latent_dim).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Cosine Annealing Scheduler will gradually reduce the LR to fine-tune the weights
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

#     print(f"Starting training on Normalized U for {epochs} epochs...")
#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0.0
        
#         for batch_u, batch_f, batch_scales in dataloader:
#             batch_u = batch_u.to(device)
            
#             u_reconstructed = model(batch_u)
#             loss = criterion(u_reconstructed, batch_u)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
            
#         # Step the scheduler at the end of every epoch
#         scheduler.step()
            
#         avg_loss = epoch_loss / len(dataloader)
        
#         if (epoch + 1) % 20 == 0 or epoch == 0:
#             current_lr = scheduler.get_last_lr()[0]
#             print(f"Epoch [{epoch+1}/{epochs}], LR: {current_lr:.6f}, MSE Loss: {avg_loss:.8f}")

#     print("\nTraining complete!")

#     # ==========================================
#     # 4. VISUALIZATION (SAVING PLOTS)
#     # ==========================================
#     print("\nGenerating reconstruction plots...")
#     model.eval()
    
#     sample_idx = 0
#     u_true_normalized = dataset[sample_idx][0].unsqueeze(0).to(device)
#     sample_scale = dataset[sample_idx][2].item()
    
#     with torch.no_grad():
#         u_pred_normalized = model(u_true_normalized)
        
#     u_true_np = u_true_normalized.cpu().numpy()[0, 0] * sample_scale
#     u_pred_np = u_pred_normalized.cpu().numpy()[0, 0] * sample_scale
    
#     slice_idx = 16 
    
#     u_true_slice = u_true_np[:, :, slice_idx]
#     u_pred_slice = u_pred_np[:, :, slice_idx]
#     error_slice = np.abs(u_true_slice - u_pred_slice)
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     im0 = axes[0].imshow(u_true_slice, cmap='viridis', origin='lower')
#     axes[0].set_title(f'Original Solution (Z={slice_idx})')
#     fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
#     im1 = axes[1].imshow(u_pred_slice, cmap='viridis', origin='lower')
#     axes[1].set_title(f'Reconstructed Solution (Z={slice_idx})')
#     fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
#     im2 = axes[2].imshow(error_slice, cmap='magma', origin='lower')
#     axes[2].set_title(f'Absolute Error (Z={slice_idx})')
#     fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
#     plt.tight_layout()
    
#     save_path = "reconstruction_plot_high_accuracy.png"
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Plot successfully saved to: {save_path}")

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. UPGRADED HIGH-CAPACITY MODEL DEFINITION
# ==========================================
class PoissonAutoencoder3D(nn.Module):
    def __init__(self, latent_dim=128): # <--- INCREASED LATENT DIMENSION
        super(PoissonAutoencoder3D, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 4, latent_dim)
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4 * 4),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            
            # <--- REMOVED TANH() SO GRADIENTS DON'T DIE AT PEAKS/TROUGHS
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 4, 4, 4) 
        return self.decoder_conv(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# ==========================================
# 2. DATA GENERATION & NORMALIZATION
# ==========================================
def generate_poisson_dataset(num_samples, grid_size=32):
    print(f"Generating {num_samples} samples for a {grid_size}^3 grid...")
    
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    z = np.linspace(0, 1, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    U_data = np.zeros((num_samples, 1, grid_size, grid_size, grid_size))
    F_data = np.zeros((num_samples, 1, grid_size, grid_size, grid_size))
    scales = np.zeros(num_samples)
    
    for i in range(num_samples):
        # High frequency parameters
        kx = np.random.uniform(1.0, 5.0)
        ky = np.random.uniform(1.0, 5.0)
        kz = np.random.uniform(1.0, 5.0)
        
        F = np.sin(2 * np.pi * kx * X) * np.sin(2 * np.pi * ky * Y) * np.sin(2 * np.pi * kz * Z)
        C = 4 * np.pi**2 * (kx**2 + ky**2 + kz**2)
        U = F / C
        
        u_max = np.max(np.abs(U))
        U_data[i, 0] = U / u_max
        F_data[i, 0] = F
        scales[i] = u_max
        
    U_tensor = torch.tensor(U_data, dtype=torch.float32)
    F_tensor = torch.tensor(F_data, dtype=torch.float32)
    scales_tensor = torch.tensor(scales, dtype=torch.float32)
    
    print(f"Per-sample normalization applied. Scale range: [{scales.min():.6f}, {scales.max():.6f}]")
    
    return TensorDataset(U_tensor, F_tensor, scales_tensor), scales_tensor

# ==========================================
# 3. SETUP & TRAINING LOOP
# ==========================================
def main():
    num_samples = 5000 # <--- INCREASED DATASET SIZE FOR HIGH FREQUENCIES
    batch_size = 64    # <--- Increased batch size for smoother gradients
    epochs = 400  
    learning_rate = 1e-3
    latent_dim = 128   # <--- WIDER LATENT SPACE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, scales = generate_poisson_dataset(num_samples=num_samples, grid_size=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PoissonAutoencoder3D(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training on Normalized U for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_u, batch_f, batch_scales in dataloader:
            batch_u = batch_u.to(device)
            
            u_reconstructed = model(batch_u)
            loss = criterion(u_reconstructed, batch_u)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
            
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{epochs}], LR: {current_lr:.6f}, MSE Loss: {avg_loss:.8f}")

    print("\nTraining complete!")

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    print("\nGenerating reconstruction plots...")
    model.eval()
    
    sample_idx = 0
    u_true_normalized = dataset[sample_idx][0].unsqueeze(0).to(device)
    sample_scale = dataset[sample_idx][2].item()
    
    with torch.no_grad():
        u_pred_normalized = model(u_true_normalized)
        
    u_true_np = u_true_normalized.cpu().numpy()[0, 0] * sample_scale
    u_pred_np = u_pred_normalized.cpu().numpy()[0, 0] * sample_scale
    
    slice_idx = 16 
    
    u_true_slice = u_true_np[:, :, slice_idx]
    u_pred_slice = u_pred_np[:, :, slice_idx]
    error_slice = np.abs(u_true_slice - u_pred_slice)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(u_true_slice, cmap='viridis', origin='lower')
    axes[0].set_title(f'Original Solution (Z={slice_idx})')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(u_pred_slice, cmap='viridis', origin='lower')
    axes[1].set_title(f'Reconstructed Solution (Z={slice_idx})')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(error_slice, cmap='magma', origin='lower')
    axes[2].set_title(f'Absolute Error (Z={slice_idx})')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    save_path = "reconstruction_plot_high_freq.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")

if __name__ == "__main__":
    main()