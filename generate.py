"""
Image Generation Script

Цей скрипт генерує нові зображення з навченої diffusion моделі.

Використання:
1. Завантажує ваги моделі
2. Запускає зворотний процес дифузії
3. Зберігає згенеровані изображения
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

# Додаємо шлях до модулів
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from model.unet import UNet
from diffusion.noise_scheduler import create_noise_scheduler
from utils.visualization import save_samples


def parse_args():
    """Парсить аргументи командного рядка."""
    parser = argparse.ArgumentParser(description='Генерація зображень з Diffusion Model')
    
    # Модель
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/latest.pt',
                       help='Шлях до чекпоінту')
    parser.add_argument('--device', type=str, default=None,
                       help='Пристрій (cuda/cpu)')
    
    # Генерація
    parser.add_argument('--num_images', type=int, default=64,
                       help='Кількість зображень для генерації')
    parser.add_argument('--image_size', type=int, default=None,
                       help='Розмір зображення')
    parser.add_argument('--channels', type=int, default=3,
                       help='Кількість каналів')
    parser.add_argument('--output', type=str, default='outputs/generated.png',
                       help='Шлях для збереження результату')
    
    # Sampler
    parser.add_argument('--sampler', type=str, default='ddim',
                       choices=['ddim', 'ddpm'],
                       help='Тип sampler')
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Кількість кроків генерації')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='Параметр стохастичності (0 = детермінований)')
    
    # Візуалізація
    parser.add_argument('--show_steps', action='store_true', help='Зберігати проміжні кроки')
    parser.add_argument('--save_gif', action='store_true', help='Створити GIF анімацію процесу')
    
    return parser.parse_args()


def step_ddim(noise_scheduler, model_output, t, sample, eta):
    """Виконує один крок DDIM алгоритму."""
    device = sample.device
    t = t.item() if hasattr(t, 'item') else t
    
    alpha_bar = noise_scheduler.alpha_bar[t].to(device)
    alpha_bar_prev = noise_scheduler.alpha_bar[t - 1].to(device) if t > 0 else torch.tensor(1.0).to(device)
    beta = noise_scheduler.beta[t].to(device)
    
    pred_original_sample = (
        sample - torch.sqrt(1 - alpha_bar) * model_output
    ) / torch.sqrt(alpha_bar)
    
    pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
    
    pred_sample_direction = torch.sqrt(1 - alpha_bar_prev - eta * beta) * model_output
    prev_sample = torch.sqrt(alpha_bar_prev) * pred_original_sample + pred_sample_direction
    
    if eta > 0 and t > 0:
        noise = torch.randn_like(sample)
        variance = torch.sqrt(eta * beta) * noise
        prev_sample = prev_sample + variance
    
    return prev_sample


def step_ddpm(noise_scheduler, model_output, t, sample, eta):
    """Крок DDPM."""
    device = sample.device
    t = t.item() if hasattr(t, 'item') else t
    
    alpha_bar_t = noise_scheduler.alpha_bar[t].to(device)
    alpha_bar_t_prev = noise_scheduler.alpha_bar[t - 1].to(device) if t > 0 else torch.tensor(1.0).to(device)
    beta_t = noise_scheduler.beta[t].to(device)
    
    pred_original_sample = (
        sample - torch.sqrt(1 - alpha_bar_t) * model_output
    ) / torch.sqrt(alpha_bar_t)\
    
    pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
    
    pred_variance = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t
    pred_variance = torch.sqrt(max(pred_variance, 0))
    
    if t > 0:
        noise = torch.randn_like(sample)
        variance = pred_variance * noise
    else:
        variance = 0
    
    pred_direction = torch.sqrt(1 - alpha_bar_t_prev - eta * pred_variance) * model_output
    prev_sample = torch.sqrt(alpha_bar_t_prev) * pred_original_sample + pred_direction + variance
    
    return prev_sample


def generate_images_ddim(model, noise_scheduler, num_images, image_size, channels, device, 
                          num_steps=None, eta=0.0, save_intermediates=False):
    """Генерує изображения за DDIM алгоритмом."""
    model.eval()
    
    shape = (num_images, channels, image_size, image_size)
    x = torch.randn(shape, device=device)
    
    intermediates = [] if save_intermediates else None
    
    if num_steps is None:
        num_steps = noise_scheduler.timesteps
    
    step_indices = torch.linspace(0, noise_scheduler.timesteps - 1, num_steps, dtype=torch.long)
    timesteps = noise_scheduler.timesteps - step_indices - 1
    
    iterator = tqdm(reversed(timesteps), total=len(timesteps), desc='DDIM Generation')
    
    for i, t in enumerate(iterator):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            noise_pred = model(x, t_batch)
        
        x = step_ddim(noise_scheduler, noise_pred, t, x, eta)
        
        if save_intermediates and (i % (len(timesteps) // 10) == 0 or i == len(timesteps) - 1):
            intermediates.append(x.clone())
    
    return x, intermediates


def generate_images_ddpm(model, noise_scheduler, num_images, image_size, channels, device, 
                          eta=1.0, save_intermediates=False):
    """Генерує изображения за DDPM алгоритмом."""
    model.eval()
    
    shape = (num_images, channels, image_size, image_size)
    x = torch.randn(shape, device=device)
    
    intermediates = [] if save_intermediates else None
    
    timesteps = torch.arange(noise_scheduler.timesteps - 1, -1, -1, device=device)
    
    for i, t in enumerate(tqdm(timesteps, desc='DDPM Generation')):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            noise_pred = model(x, t_batch)
        
        x = step_ddpm(noise_scheduler, noise_pred, t, x, eta)
        
        if save_intermediates and (i % (noise_scheduler.timesteps // 10) == 0):
            intermediates.append(x.clone())
    
    return x, intermediates


def main():
    """Головна функція генерації."""
    
    args = parse_args()
    
    if args.device is not None:
        cfg.DEVICE = args.device
    if args.image_size is not None:
        cfg.IMAGE_SIZE = args.image_size
    
    device = cfg.DEVICE
    print(f"Using device: {device}")
    
    checkpoint_path = args.checkpoint
    
    if not os.path.exists(checkpoint_path):
        alt_paths = [
            checkpoint_path,
            os.path.join(cfg.CHECKPOINT_DIR, 'latest.pt'),
            os.path.join(cfg.CHECKPOINT_DIR, 'final_model.pt'),
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        else:
            print(f"Error: Checkpoint not found!")
            print(f"Available: {os.listdir(cfg.CHECKPOINT_DIR) if os.path.exists(cfg.CHECKPOINT_DIR) else 'dir does not exist'}")
            return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_config = checkpoint.get('config', cfg.get_config())
    
    image_size = args.image_size or model_config.get('image_size', cfg.IMAGE_SIZE)
    channels = args.channels
    timesteps = model_config.get('timesteps', cfg.TIMESTEPS)
    base_channels = model_config.get('base_channels', cfg.BASE_CHANNELS)
    channel_multiplier = model_config.get('channels_multiplier', cfg.CHANNELS_MULTIPLIER)
    num_res_blocks = model_config.get('num_res_blocks', cfg.NUM_RES_BLOCKS)
    time_embed_dim = model_config.get('time_embed_dim', cfg.TIME_EMBED_DIM)
    dropout = model_config.get('dropout', cfg.DROPOUT)
    
    print(f"Model config:")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Channels: {channels}")
    print(f"  - Timesteps: {timesteps}")
    print(f"  - Base channels: {base_channels}")
    
    print("\nCreating model...")
    model = UNet(
        in_channels=channels,
        out_channels=channels,
        base_channels=base_channels,
        channel_multiplier=channel_multiplier,
        num_res_blocks=num_res_blocks,
        time_embed_dim=time_embed_dim,
        dropout=dropout,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded!")
    
    print("Creating noise scheduler...")
    noise_scheduler = create_noise_scheduler(
        timesteps=timesteps,
        beta_start=model_config.get('beta_start', cfg.BETA_START),
        beta_end=model_config.get('beta_end', cfg.BETA_END),
        schedule=model_config.get('beta_schedule', cfg.BETA_SCHEDULE),
    )
    
    print(f"\nGenerating {args.num_images} images...")
    
    if args.sampler == 'ddim':
        samples, intermediates = generate_images_ddim(
            model, noise_scheduler, args.num_images, image_size, channels, device,
            num_steps=args.num_steps, eta=args.eta,
            save_intermediates=args.show_steps or args.save_gif,
        )
    else:
        samples, intermediates = generate_images_ddpm(
            model, noise_scheduler, args.num_images, image_size, channels, device,
            eta=args.eta, save_intermediates=args.show_steps or args.save_gif,
        )
    
    print(f"\nSaving result: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_samples(samples, args.output, nrow=8)
    
    if args.show_steps and intermediates:
        intermediates_path = args.output.replace('.png', '_steps.png')
        from utils.visualization import save_diffusion_process
        save_diffusion_process(intermediates, intermediates_path)
        print(f"Intermediate steps saved: {intermediates_path}")
    
    if args.save_gif and intermediates:
        gif_path = args.output.replace('.png', '_process.gif')
        from utils.visualization import create_gif_from_images
        create_gif_from_images(intermediates, gif_path)
    
    print("\nGeneration complete!")
    print(f"Result: {args.output}")


if __name__ == '__main__':
    main()
