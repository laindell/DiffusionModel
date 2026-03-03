"""
Reverse Diffusion Sampler

Цей модуль реалізує зворотний процес (reverse process) для генерації
зображень з шуму.

Reverse Process (p):
p(x_{t-1} | x_t) = N(x_{t-1}; mu(x_t, t), sigma_t^2 * I)

де:
- mu(x_t, t): передбачене середнє значення (обчислюється з моделі)
- sigma_t: дисперсія шуму

Алгоритм DDPM:
1. Починаємо з чистого гаусівського шуму x_T
2. Для t = T, T-1, ..., 1:
   - Передбачаємо шум ε за допомогою моделі
   - Обчислюємо x_{t-1}
3. Отримуємо x_0 - згенероване зображення
"""

import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
from .noise_scheduler import NoiseScheduler


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.
    
    DDIM є більш швидким варіантом DDPM, який використовує
    детермінований шлях для генерації.
    
    Основна відмінність від DDPM:
    - DDPM: стохастичний (додає випадковий шум на кожному кроці)
    - DDIM: детермінований (або напів-детермінований)
    
    Args:
        noise_scheduler: Планувальник шуму
    """
    
    def __init__(self, noise_scheduler: NoiseScheduler):
        self.noise_scheduler = noise_scheduler
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        guidance_scale: Optional[float] = None,
        condition: Optional[torch.Tensor] = None,
        device: str = 'cpu',
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Генерує зображення за допомогою DDIM алгоритму.
        
        Args:
            model: Навчена U-Net модель
            shape: Форма вихідного тензора (batch, channels, height, width)
            num_steps: Кількість кроків генерації (менше = швидше)
            eta: Параметр стохастичності (0 = повністю детермінований)
            guidance_scale: Scale для classifier-free guidance (не використовується тут)
            condition: Умовна інформація (для умовної генерації)
            device: Пристрій
            verbose: Показувати прогрес
            
        Returns:
            Згенеровані зображення
        """
        batch_size, channels, height, width = shape
        
        # Використовуємо всі timesteps якщо не вказано
        if num_steps is None:
            num_steps = self.noise_scheduler.timesteps
        
        # Створюємо скорочений набір timesteps
        step_indices = torch.linspace(0, self.noise_scheduler.timesteps - 1, num_steps, dtype=torch.long)
        timesteps = self.noise_scheduler.timesteps - step_indices - 1
        
        # Починаємо з випадкового шуму
        img = torch.randn(shape, device=device)
        
        # Зберігаємо проміжні результати для візуалізації
        intermediates = [] if verbose else None
        
        # Ітеративно видаляємо шум
        iterator = tqdm(reversed(timesteps), total=len(timesteps), disable=not verbose)
        
        for i, t in enumerate(iterator):
            # Створюємо тензор timestep для батчу
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Передбачаємо шум
            noise_pred = model(img, t_batch)
            
            # Обчислюємо попереднє зображення
            img = self._step(noise_pred, t, img, eta)
            
            # Зберігаємо проміжний результат
            if verbose and (i % (len(timesteps) // 10) == 0 or i == len(timesteps) - 1):
                intermediates.append(img.clone())
        
        return img, intermediates
    
    def _step(
        self,
        model_output: torch.Tensor,
        t: int,
        sample: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        """
        Виконує один крок DDIM алгоритму.
        
        Args:
            model_output: Передбачений шум
            t: Поточний timestep
            sample: Поточне зображення
            eta: Параметр стохастичності
            
        Returns:
            Попереднє зображення
        """
        device = sample.device
        t = t.item() if hasattr(t, 'item') else t
        
        # Отримуємо значення з планувальника
        alpha_bar = self.noise_scheduler.alpha_bar[t].to(device)
        alpha_bar_prev = self.noise_scheduler.alpha_bar[t - 1].to(device) if t > 0 else torch.tensor(1.0).to(device)
        beta = self.noise_scheduler.beta[t].to(device)
        
        # Обчислюємо передбачене оригінальне зображення
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_bar) * model_output
        ) / torch.sqrt(alpha_bar)
        
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        # Обчислюємо напрямок до попереднього
        pred_sample_direction = torch.sqrt(1 - alpha_bar_prev - eta * beta) * model_output
        
        # Обчислюємо попереднє зображення
        prev_sample = torch.sqrt(alpha_bar_prev) * pred_original_sample + pred_sample_direction
        
        # Додаємо стохастичність якщо eta > 0
        if eta > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(eta * beta) * noise
            prev_sample = prev_sample + variance
        
        return prev_sample


class DDPMSampler:
    """
    DDPM sampler (оригінальний алгоритм).
    
    Цей sampler використовує стохастичний підхід з випадковим
    шумом на кожному кроці.
    
    Args:
        noise_scheduler: Планувальник шуму
    """
    
    def __init__(self, noise_scheduler: NoiseScheduler):
        self.noise_scheduler = noise_scheduler
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        num_steps: Optional[int] = None,
        eta: float = 1.0,
        device: str = 'cpu',
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Генерує зображення за допомогою DDPM алгоритму.
        
        Args:
            model: Навчена U-Net модель
            shape: Форма вихідного тензора
            num_steps: Кількість кроків (якщо None - використовуємо повний набір)
            eta: Параметр стохастичності (1 = оригінальний DDPM)
            device: Пристрій
            verbose: Показувати прогрес
            
        Returns:
            Згенеровані зображення
        """
        batch_size, channels, height, width = shape
        
        # Починаємо з випадкового шуму
        img = torch.randn(shape, device=device)
        
        # Створюємо timesteps
        timesteps = torch.arange(self.noise_scheduler.timesteps - 1, -1, -1, device=device)
        
        # Зберігаємо проміжні результати
        intermediates = [] if verbose else None
        
        # Ітеративно видаляємо шум
        iterator = tqdm(timesteps, disable=not verbose)
        
        for i, t in enumerate(iterator):
            # Створюємо тензор timestep
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Передбачаємо шум
            noise_pred = model(img, t_batch)
            
            # Обчислюємо попереднє зображення
            img = self._step_ddpm(noise_pred, t, img, eta)
            
            # Зберігаємо проміжний результат
            if verbose and (i % (self.noise_scheduler.timesteps // 10) == 0):
                intermediates.append(img.clone())
        
        return img, intermediates
    
    def _step_ddpm(
        self,
        model_output: torch.Tensor,
        t: int,
        sample: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        """
        Виконує один крок DDPM алгоритму.
        
        Формула:
        x_{t-1} = (x_t - (1-ᾱ_t)/√(1-ᾱ_t) * ε_θ(x_t,t)) / √α_t + σ_t * z
        
        Args:
            model_output: Передбачений шум
            t: Поточний timestep
            sample: Поточне зображення
            eta: Параметр стохастичності
            
        Returns:
            Попереднє зображення
        """
        device = sample.device
        t = t.item() if hasattr(t, 'item') else t
        
        # Отримуємо значення
        alpha_bar_t = self.noise_scheduler.alpha_bar[t].to(device)
        alpha_bar_t_prev = self.noise_scheduler.alpha_bar[t - 1].to(device) if t > 0 else torch.tensor(1.0).to(device)
        beta_t = self.noise_scheduler.beta[t].to(device)
        
        # Обчислюємо передбачене оригінальне зображення
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_bar_t) * model_output
        ) / torch.sqrt(alpha_bar_t)
        
        # Обчислюємо коефіцієнт для шуму
        pred_original_sample_variance = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t
        pred_original_sample_variance = torch.sqrt(max(pred_original_sample_variance, 0))
        
        # Обчислюємо попереднє зображення
        if t > 0:
            noise = torch.randn_like(sample)
            variance = pred_original_sample_variance * noise
        else:
            variance = 0
        
        # Головний член
        pred_sample_direction = torch.sqrt(1 - alpha_bar_t_prev - eta * pred_original_sample_variance) * model_output
        
        prev_sample = torch.sqrt(alpha_bar_t_prev) * pred_original_sample + pred_sample_direction + variance
        
        return prev_sample


def create_sampler(
    noise_scheduler: NoiseScheduler,
    sampler_type: str = 'ddim',
) -> DDIMSampler:
    """
    Фабрична функція для створення sampler.
    
    Args:
        noise_scheduler: Планувальник шуму
        sampler_type: Тип sampler ('ddim' або 'ddpm')
        
    Returns:
        Ініціалізований sampler
    """
    if sampler_type.lower() == 'ddim':
        return DDIMSampler(noise_scheduler)
    elif sampler_type.lower() == 'ddpm':
        return DDPMSampler(noise_scheduler)
    else:
        raise ValueError(f"Невідомий тип sampler: {sampler_type}")
