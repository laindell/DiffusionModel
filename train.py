"""
Training Script for Diffusion Model

Цей скрипт виконує навчання DDPM (Denoising Diffusion Probabilistic Model).

Процес навчання:
1. Завантажуємо датасет
2. На кожному кроці:
   a. Вибираємо випадковий timestep t
   b. Генеруємо випадковий шум ε
   c. Створюємо зашумлене изображение x_t
   d. Передбачаємо шум за допомогою U-Net
   e. Обчислюємо MSE loss
   f. Оновлюємо ваги
3. Зберігаємо чекпоінти та генеруємо приклади
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Додаємо шлях до модулів
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Імпортуємо модулі проєкту
import config as cfg
from model.unet import UNet
from model.timestep_embedding import SinusoidalPositionEmbeddings
from diffusion.noise_scheduler import create_noise_scheduler
from diffusion.forward_diffusion import ForwardDiffusion
from data.dataset_loader import load_dataset
from utils.checkpoint import CheckpointManager
from utils.visualization import (
    save_samples, 
    plot_loss_curve, 
    plot_noise_schedule,
    TrainingVisualizer,
    denormalize
)


def parse_args():
    """Парсить аргументи командного рядка."""
    parser = argparse.ArgumentParser(description='Навчання Diffusion Model')
    
    # Датасет
    parser.add_argument('--dataset', type=str, default=None,
                       help='Назва датасету (mnist, cifar10, folder)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Директорія для даних')
    
    # Навчання
    parser.add_argument('--epochs', type=int, default=None,
                       help='Кількість епох')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Розмір батчу')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=None,
                       help='Розмір зображення')
    
    # Модель
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Кількість timesteps')
    parser.add_argument('--base_channels', type=int, default=None,
                       help='Базові канали U-Net')
    
    # Різне
    parser.add_argument('--device', type=str, default=None,
                       help='Пристрій (cuda/cpu)')
    parser.add_argument('--resume', action='store_true',
                       help='Продовжити навчання з останнього чекпоінту')
    
    return parser.parse_args()


def train_step(
    model: nn.Module,
    forward_diffusion: ForwardDiffusion,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    device: str,
    scaler: GradScaler = None,
) -> float:
    """
    Виконує один крок навчання.
    
    Args:
        model: U-Net модель
        forward_diffusion: Модуль прямого процесу
        optimizer: Оптимізатор
        images: Батч зображень
        device: Пристрій
        scaler: Gradient scaler для mixed precision
        
    Returns:
        Значення loss
    """
    model.train()
    
    # Переносимо зображення на пристрій
    images = images.to(device)
    
    # Генеруємо випадкові timesteps
    t = forward_diffusion.sample_timesteps(images.shape[0], device)
    
    # Генеруємо шум
    noise = torch.randn_like(images)
    
    # Створюємо зашумлене изображение
    noisy_images, _ = forward_diffusion.add_noise(images, noise, t)
    
    # Передбачаємо шум
    if scaler is not None:
        with autocast():
            noise_pred = model(noisy_images, t)
            loss = F.mse_loss(noise_pred, noise)
        
        # Оновлюємо з mixed precision
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else:
        noise_pred = model(noisy_images, t)
        loss = F.mse_loss(noise_pred, noise)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
        optimizer.step()
    
    optimizer.zero_grad()
    
    return loss.item()


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    noise_scheduler,
    device: str,
    num_samples: int = 16,
    image_size: int = 32,
    channels: int = 3,
) -> torch.Tensor:
    """
    Генерує зразки для візуалізації.
    
    Args:
        model: Модель
        noise_scheduler: Планувальник шуму
        device: Пристрій
        num_samples: Кількість зразків
        image_size: Розмір зображення
        channels: Кількість каналів
        
    Returns:
        Згенеровані зображення
    """
    model.eval()
    
    # Починаємо з шуму
    shape = (num_samples, channels, image_size, image_size)
    x = torch.randn(shape, device=device)
    
    # Ітеруємо через усі timesteps
    for t in tqdm(range(noise_scheduler.timesteps - 1, -1, -1), desc='Sampling'):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        # Передбачаємо шум
        noise_pred = model(x, t_batch)
        
        # Обчислюємо попереднє зображення
        x = noise_scheduler.step(noise_pred, t, x)
    
    return x


def main():
    """Головна функція навчання."""
    
    # Парсимо аргументи
    args = parse_args()
    
    # Перевизначаємо конфігурацію якщо вказані аргументи
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.LEARNING_RATE = args.lr
    if args.image_size is not None:
        cfg.IMAGE_SIZE = args.image_size
    if args.timesteps is not None:
        cfg.TIMESTEPS = args.timesteps
    if args.base_channels is not None:
        cfg.BASE_CHANNELS = args.base_channels
    if args.device is not None:
        cfg.DEVICE = args.device
    if args.dataset is not None:
        cfg.DATASET_PATH = args.dataset
    
    # Створюємо директорії
    cfg.create_directories()
    
    device = cfg.DEVICE
    print(f"Використовуємо пристрій: {device}")
    
    # Завантажуємо датасет
    print(f"Завантажуємо датасет: {cfg.DATASET_PATH}")
    dataloader, num_channels = load_dataset(
        cfg.DATASET_PATH,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        is_training=True,
        data_dir=args.data_dir,
    )
    print(f"Каналів у датасеті: {num_channels}")
    
    # Створюємо модель
    print("Створюємо U-Net модель...")
    model = UNet(
        in_channels=num_channels,
        out_channels=num_channels,
        base_channels=cfg.BASE_CHANNELS,
        channel_multiplier=cfg.CHANNELS_MULTIPLIER,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        time_embed_dim=cfg.TIME_EMBED_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)
    
    print(f"Кількість параметрів: {model.get_num_parameters():,}")
    
    # Створюємо optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    
    # Створюємо scheduler для шуму
    print("Ініціалізуємо Noise Scheduler...")
    noise_scheduler = create_noise_scheduler(
        timesteps=cfg.TIMESTEPS,
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END,
        schedule=cfg.BETA_SCHEDULE,
    )
    
    # Forward diffusion
    forward_diffusion = ForwardDiffusion(noise_scheduler)
    
    # Gradient scaler для mixed precision
    scaler = None
    if cfg.USE_MIXED_PRECISION and device == 'cuda':
        scaler = GradScaler()
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(cfg.CHECKPOINT_DIR)
    
    # Візуалізатор
    visualizer = TrainingVisualizer(cfg.SAMPLES_DIR)
    
    # Візуалізуємо noise schedule
    noise_schedule_path = os.path.join(cfg.OUTPUT_DIR, 'noise_schedule.png')
    plot_noise_schedule(
        noise_scheduler.beta,
        noise_scheduler.alpha,
        noise_scheduler.alpha_bar,
        noise_schedule_path
    )
    
    # Продовжуємо навчання якщо вказано
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = checkpoint_manager.load_checkpoint(model, optimizer)
            start_epoch = checkpoint['epoch'] + 1
            print(f"Продовжуємо з епохи {start_epoch}")
        except FileNotFoundError:
            print("Не знайдено чекпоінт. Починаємо з нуля.")
    
    # Цикл навчання
    print(f"\nПочинаємо навчання на {cfg.EPOCHS} епох...")
    
    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_losses = []
        
        # tqdm для прогресу
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{cfg.EPOCHS}')
        
        for batch_idx, images in enumerate(pbar):
            # Крок навчання
            loss = train_step(
                model,
                forward_diffusion,
                optimizer,
                images,
                device,
                scaler,
            )
            
            # Зберігаємо loss
            epoch_losses.append(loss)
            visualizer.add_loss(loss)
            
            # Оновлюємо tqdm
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # Середній loss за епоху
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} - Loss: {avg_loss:.4f}")
        
        # Зберігаємо чекпоінт
        if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
            checkpoint_manager.save_checkpoint(
                model,
                optimizer,
                epoch,
                avg_loss,
                cfg.get_config(),
            )
        
        # Візуалізуємо результати
        if (epoch + 1) % cfg.VISUALIZE_EVERY == 0:
            # Генеруємо зразки
            print("Генеруємо зразки...")
            samples = generate_samples(
                model,
                noise_scheduler,
                device,
                num_samples=16,
                image_size=cfg.IMAGE_SIZE,
                channels=num_channels,
            )
            
            # Зберігаємо зразки
            sample_path = os.path.join(cfg.SAMPLES_DIR, f'samples_epoch_{epoch+1:04d}.png')
            save_samples(samples, sample_path, nrow=4)
            
            # Зберігаємо loss plot
            visualizer.save_loss_plot()
    
    # Фінальний чекпоінт
    print("\nЗберігаємо фінальний чекпоінт...")
    checkpoint_manager.save_checkpoint(
        model,
        optimizer,
        cfg.EPOCHS - 1,
        avg_loss,
        cfg.get_config(),
        filename='final_model.pt',
    )
    
    # Фінальна генерація
    print("Генеруємо фінальні зразки...")
    samples = generate_samples(
        model,
        noise_scheduler,
        device,
        num_samples=64,
        image_size=cfg.IMAGE_SIZE,
        channels=num_channels,
    )
    
    sample_path = os.path.join(cfg.SAMPLES_DIR, 'final_samples.png')
    save_samples(samples, sample_path, nrow=8)
    
    print("\nНавчання завершено!")
    print(f"Результати збережено в: {cfg.OUTPUT_DIR}")


if __name__ == '__main__':
    main()