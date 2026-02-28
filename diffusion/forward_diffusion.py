"""
Forward Diffusion Process

Цей модуль реалізує прямий процес дифузії (forward process), який
поступово додає гаусівський шум до зображення.

Forward Process (q):
q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

Це означає, що ми можемо додати шум за один крок:
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

де epsilon ~ N(0, I)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .noise_scheduler import NoiseScheduler


class ForwardDiffusion(nn.Module):
    """
    Модуль для виконання прямого процесу дифузії.
    
    Основні функції:
    - Генерація випадкових timesteps
    - Додавання шуму до зображень
    - Візуалізація процесу забруднення
    """
    
    def __init__(self, noise_scheduler: NoiseScheduler):
        super().__init__()
        self.noise_scheduler = noise_scheduler
    
    def sample_timesteps(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """
        Випадково вибирає timesteps для батчу.
        
        Args:
            batch_size: Розмір батчу
            device: Пристрій
            
        Returns:
            Tensor з випадковими timesteps
        """
        return torch.randint(
            0, 
            self.noise_scheduler.timesteps, 
            (batch_size,), 
            device=device,
            dtype=torch.long
        )
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Додає шум до зображень для навчання.
        
        Це основна функція, яка використовується під час навчання:
        1. Вибираємо випадкові timesteps
        2. Генеруємо випадковий шум
        3. Обчислюємо зашумлене зображення
        
        Args:
            x_start: Оригінальні зображення (batch, channels, height, width)
            noise: Випадковий шум (якщо None - генерується)
            t: Timesteps (якщо None - вибираються випадково)
            
        Returns:
            Tuple з (зашумлене_зображення, шум)
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Якщо t не вказані, вибираємо випадкові
        if t is None:
            t = self.sample_timesteps(batch_size, device)
        
        # Якщо шум не вказаний, генеруємо
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Додаємо шум використовуючи noise_scheduler
        x_t = self.noise_scheduler.add_noise(x_start, noise, t)
        
        return x_t, noise
    
    def get_noisy_image(
        self,
        x_start: torch.Tensor,
        t: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Отримує зашумлене зображення на конкретному timestep.
        
        Args:
            x_start: Оригінальне зображення
            t: Конкретний timestep
            noise: Шум (опціонально)
            
        Returns:
            Зашумлене зображення
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Створюємо тензор timestep
        t_tensor = torch.full(
            (x_start.shape[0],), 
            t, 
            device=x_start.device, 
            dtype=torch.long
        )
        
        return self.noise_scheduler.add_noise(x_start, noise, t_tensor)
    
    def visualize_diffusion(
        self,
        x_start: torch.Tensor,
        num_images: int = 10,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Створює візуалізацію процесу забруднення.
        
        Показує, як зображення поступово перетворюється на шум.
        
        Args:
            x_start: Оригінальне зображення (1, channels, H, W)
            num_images: Кількість зображень для візуалізації
            device: Пристрій
            
        Returns:
            Tensor з изображениями на різних кроках
        """
        steps = torch.linspace(0, self.noise_scheduler.timesteps - 1, num_images, dtype=torch.long)
        images = []
        
        noise = torch.randn_like(x_start)
        
        for t in steps:
            t_tensor = torch.full((1,), t.item(), device=device, dtype=torch.long)
            noisy_img = self.noise_scheduler.add_noise(x_start, noise, t_tensor)
            images.append(noisy_img)
        
        return torch.cat(images, dim=0)
    
    def sample_random_timesteps(
        self,
        batch_size: int,
        num_timesteps: Optional[int] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Вибирає випадкові timesteps з рівномірним розподілом.
        
        Args:
            batch_size: Розмір батчу
            num_timesteps: Скільки unique timesteps вибрати
            device: Пристрій
            
        Returns:
            Tensor з timesteps
        """
        if num_timesteps is None:
            num_timesteps = self.noise_scheduler.timesteps
        
        # Вибираємо випадкові індекси
        t = torch.randint(
            0,
            num_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long
        )
        
        return t


def create_forward_diffusion(
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    schedule: str = 'linear',
    device: str = 'cpu'
) -> ForwardDiffusion:
    """
    Фабрична функція для створення ForwardDiffusion.
    
    Args:
        timesteps: Кількість кроків дифузії
        beta_start: Початкове значення beta
        beta_end: Кінцеве значення beta
        schedule: Тип розкладу
        device: Пристрій
        
    Returns:
        Ініціалізований ForwardDiffusion
    """
    from .noise_scheduler import create_noise_scheduler
    
    noise_scheduler = create_noise_scheduler(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule=schedule,
        device=device
    )
    
    return ForwardDiffusion(noise_scheduler)
