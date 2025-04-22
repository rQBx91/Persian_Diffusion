import torch
from torch import nn
import torch.nn.functional as F
from dataset_loader_persian import data_loader
from unet import UNet
import numpy as np
from torch.optim import Adam
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import matplotlib.animation as animation
import time


def get_linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()) 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)        


class DiffusionModel(nn.Module):
    
    def __init__(self, reverse_process_model, beta_start, beta_end, timesteps=500, device="cuda"):
        super().__init__()
        self.reverse_process_model = reverse_process_model
        self.timesteps = timesteps

        # Define beta schedule
        self.betas = get_linear_beta_schedule(beta_start, beta_end, timesteps)
        
        # Define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)


        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    

    def get_loss(self, reverse_process_model, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
    
        x_noisy = self.add_noise(x_0=x_0, t=t, noise=noise)
        predicted_noise = reverse_process_model(x_noisy, t)
    
        loss = F.mse_loss(noise, predicted_noise)
    
        return loss

    @torch.no_grad()
    def p_sample(self, reverse_process_model, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)


        # Use our network (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * reverse_process_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            
            return model_mean + torch.sqrt(posterior_variance_t) * noise 


    @torch.no_grad()
    def p_sample_loop(self, reverse_process_model, shape):
        device = next(reverse_process_model.parameters()).device

        b = shape[0]
        # Start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling...', total=self.timesteps):
            img = self.p_sample(reverse_process_model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu())
        return imgs


    @torch.no_grad()
    def sample(self, reverse_process_model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(reverse_process_model, shape=(batch_size, channels, image_size, image_size))


def train(diffusion_model, dataloader, optimizer, epochs, reverse_process_model, image_size, device="cuda"):
    # Keep loss history
    loss_history = []
    for epoch in range(epochs):
        t0 = time.time()
        print(f'epoch {epoch+1}/{epochs}')
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Batch information
            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Sample t uniformally for every example in the batch
            t = torch.randint(0, diffusion_model.timesteps, (batch_size,), device=device).long()

            # Calculate loss
            loss = diffusion_model.get_loss(reverse_process_model, batch, t)

            # Monitor loss 
            if step % 100 == 0:
                print("Loss:", loss.item())
                loss_history.append(loss.item())

            loss.backward()
            optimizer.step()

        # Check training progress
        all_images_list = list(map(lambda n: diffusion_model.sample(reverse_process_model,image_size=image_size, batch_size=n, channels=channels), [8]))
        all_images = torch.cat(all_images_list[0], dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(train_folder / f'epoch{epoch+1}.png'), nrow = 8)

        # Save model state and weights
        state = {
        'epoch': epoch+1,
        'model_state_dict': diffusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, f'{model_folder}/diffusion_model_epoch_{epoch+1}.pth')

        # Elapsed time of epoch
        t1 = time.time()
        elapsed = t1 - t0
        print(f'epoch {epoch+1} took {elapsed:.2f} seconds')

    # Plot loss
    y = loss_history
    x = np.linspace(start=1, stop=epochs, num=len(loss_history))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('loss curve persian')
    plt.plot(x, y)
    plt.savefig('loss_curve_persian.png')
    

if __name__ == "__main__":
    # Define mode: 1 for training 0 for inference
    #mode = 1 # training
    mode = 0 # inference

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device}...')

    # Hyperparameters
    batch_size = 64
    epochs = 50
    learning_rate = 0.001
    beta_start = 1e-4
    beta_end = 2e-2

    # Image description
    image_size = 32
    channels = 1

    # Paths and directories
    results_folder = Path("./result/")
    results_folder.mkdir(exist_ok = True)

    results_folder = Path("./result/persian/")
    results_folder.mkdir(exist_ok = True)

    train_folder = Path("./result/persian/train")
    train_folder.mkdir(exist_ok = True)

    samples_folder = Path("./result/persian/samples/")
    samples_folder.mkdir(exist_ok = True)

    images_folder = Path("./result/persian/samples/images/")
    images_folder.mkdir(exist_ok = True)

    gifs_folder = Path("./result/persian/samples/gifs/")
    gifs_folder.mkdir(exist_ok = True)

    model_folder = Path("./result/persian/model")
    model_folder.mkdir(exist_ok = True)

    # Load persian dataset
    train_loader = data_loader(batch_size)

    
    # Define UNet model
    reverse_process_model = UNet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
    )
    reverse_process_model.to(device)

    # Define optimizer
    optimizer = Adam(reverse_process_model.parameters(), lr=learning_rate)
    # Define diffiusion model
    diffusion_model = DiffusionModel(reverse_process_model, beta_start, beta_end).to(device)
    
    # Inference mode
    if mode == 0:
        # Load the model
        checkpoint = torch.load(f'{model_folder}/diffusion_model_epoch_50.pth')
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Treaining mode
    if mode == 1:
        # Train the model
        train(diffusion_model, train_loader, optimizer, epochs, reverse_process_model, image_size, device)

    
    # Sample from model
    sample_num = 32
    samples = diffusion_model.sample(reverse_process_model, image_size=image_size, batch_size=sample_num, channels=channels)

    # Save samples
    for index in range(sample_num):
        # Save sample as png
        plt.imshow(samples[-1][index].reshape(image_size, image_size, channels), cmap="gray")
        plt.savefig(f'{images_folder}/image{index+1}.png')

        # Make a gif
        fig = plt.figure()
        ims = []
        for i in range(diffusion_model.timesteps):
            im = plt.imshow(samples[i][index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
            ims.append([im])

        animate = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=5000)
        animate.save(f'{gifs_folder}/gif{index+1}.gif')
        plt.close(fig)

