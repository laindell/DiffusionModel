"""
Visualization Utilities

Цей модуль містить функції для візуалізації:
- Зображення з датасету
- Процесу дифузії (від чистого до шуму)
- Результатів генерації
- Графіків loss
"""

import os
import torch
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """
    Денормалізує изображения з [-1, 1] до [0, 1].
    
    Args:
        images: Тензор зображень
        
    Returns:
        Денормалізовані зображення
    """
    return (images + 1) / 2


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    normalize: bool = True,
    padding: int = 2,
) -> None:
    """
    Зберігає сітку зображень.
    
    Args:
        images: Тензор (batch, channels, height, width)
        path: Шлях для збереження
        nrow: Кількість зображень в рядку
        normalize: Чи нормалізувати перед збереженням
        padding: Відступи між изображениями
    """
    if normalize:
        images = denormalize(images)
    
    # Обрізаємо до [0, 1]
    images = torch.clamp(images, 0, 1)
    
    # Створюємо сітку
    grid = vutils.make_grid(images, nrow=nrow, padding=padding, normalize=False)
    
    # Зберігаємо
    vutils.save_image(grid, path)


def save_samples(
    samples: torch.Tensor,
    path: str,
    nrow: int = 8,
) -> None:
    """
    Зберігає згенеровані зразки.
    
    Args:
        samples: Згенеровані зображення
        path: Шлях для збереження
        nrow: Кількість в рядку
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image_grid(samples, path, nrow=nrow)


def save_diffusion_process(
    images: List[torch.Tensor],
    path: str,
    nrow: int = 10,
) -> None:
    """
    Зберігає візуалізацію процесу дифузії.
    
    Показує зображення на різних кроках процесу.
    
    Args:
        images: Список зображень на різних кроках
        path: Шлях для збереження
        nrow: Кількість в рядку
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Конвертуємо список в тензор
    images_tensor = torch.stack(images)
    
    # Денормалізуємо
    images_tensor = denormalize(images_tensor)
    images_tensor = torch.clamp(images_tensor, 0, 1)
    
    # Зберігаємо
    save_image_grid(images_tensor, path, nrow=nrow)


def plot_loss_curve(
    losses: List[float],
    path: str,
    window_size: int = 100,
) -> None:
    """
    Зберігає графік кривої втрат.
    
    Args:
        losses: Список значень loss
        path: Шлях для збереження
        window_size: Розмір вікна для ковзного середнього
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    # Графік raw loss
    plt.plot(losses, alpha=0.3, label='Loss', linewidth=0.5)
    
    # Ковзне середнє
    if len(losses) > window_size:
        losses_smooth = []
        for i in range(len(losses) - window_size + 1):
            losses_smooth.append(np.mean(losses[i:i+window_size]))
        plt.plot(range(window_size - 1, len(losses)), losses_smooth, 
                 label=f'Smoothed (window={window_size})', linewidth=2)
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_noise_schedule(
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    path: str,
) -> None:
    """
    Візуалізує розклад шуму (beta, alpha, alpha_bar).
    
    Args:
        betas: Значення beta
        alphas: Значення alpha
        alpha_bars: Значення alpha_bar
        path: Шлях для збереження
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    t = range(len(betas))
    
    # Beta
    axes[0].plot(t, betas.numpy())
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('β (beta)')
    axes[0].set_title('Beta Schedule')
    axes[0].grid(True, alpha=0.3)
    
    # Alpha
    axes[1].plot(t, alphas.numpy())
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('α (alpha)')
    axes[1].set_title('Alpha Schedule')
    axes[1].grid(True, alpha=0.3)
    
    # Alpha bar
    axes[2].plot(t, alpha_bars.numpy())
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('ᾱ (alpha_bar)')
    axes[2].set_title('Alpha Bar (Cumulative Product)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def visualize_batch(
    batch: torch.Tensor,
    path: str,
    nrow: int = 8,
    title: str = 'Batch Samples',
) -> None:
    """
    Візуалізує батч зображень.
    
    Args:
        batch: Батч зображень
        path: Шлях для збереження
        nrow: Кількість в рядку
        title: Заголовок
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    plt.figure(figsize=(12, 12))
    save_image_grid(batch, path, nrow=nrow)
    plt.close()


def create_gif_from_images(
    images: List[torch.Tensor],
    path: str,
    duration: float = 0.5,
) -> None:
    """
    Створює GIF анімацію з послідовності зображень.
    
    Args:
        images: Список зображень
        path: Шлях для збереження
        duration: Тривалість кожного кадру в секундах
    """
    try:
        from PIL import Image
        import numpy as np
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Денормалізуємо та конвертуємо в PIL Images
        pil_images = []
        for img in images:
            img = denormalize(img)
            img = torch.clamp(img, 0, 1)
            
            # Конвертуємо в numpy
            img_np = img.cpu().numpy()
            if img_np.shape[0] == 3:  # CHW -> HWC
                img_np = np.transpose(img_np, (1, 2, 0))
            
            # Конвертуємо в PIL
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
            pil_images.append(pil_img)
        
        # Зберігаємо як GIF
        pil_images[0].save(
            path,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(duration * 1000),
            loop=0
        )
        
        print(f"GIF збережено: {path}")
        
    except ImportError:
        print("PIL не встановлено. Пропускаємо створення GIF.")


class TrainingVisualizer:
    """
    Клас для візуалізації процесу навчання.
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        self.losses = []
        os.makedirs(output_dir, exist_ok=True)
    
    def add_loss(self, loss: float):
        """Додає значення loss."""
        self.losses.append(loss)
    
    def save_samples(
        self,
        samples: torch.Tensor,
        epoch: int,
        prefix: str = 'samples',
    ):
        """Зберігає зразки зображень."""
        path = os.path.join(self.output_dir, f'{prefix}_epoch_{epoch:04d}.png')
        save_samples(samples, path)
    
    def save_loss_plot(self, path: Optional[str] = None):
        """Зберігає графік loss."""
        if path is None:
            path = os.path.join(self.output_dir, 'loss.png')
        plot_loss_curve(self.losses, path)
    
    def get_average_loss(self, window: int = 100) -> float:
        """Повертає середній loss за останні window кроків."""
        if not self.losses:
            return 0.0
        return np.mean(self.losses[-window:])
