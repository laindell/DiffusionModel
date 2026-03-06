"""
Noise Scheduler (DDPM) - Планувальник шуму

Цей модуль реалізує математичну основу DDPM (Denoising Diffusion Probabilistic Models).

Основні концепції:
- beta: дисперсія шуму на кожному кроці t
- alpha: 1 - beta
- alpha_hat: кумулятивний добуток alpha від 0 до t

Forward Process (q - пряма дифузія):
x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * epsilon

де:
- x_0: оригінальне зображення
- x_t: зашумлене зображення на кроці t
- epsilon: випадковий гаусівський шум
"""

import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    """
    Планувальник шуму для DDPM.
    
    Обчислює всі необхідні коефіцієнти для прямого та зворотного процесів.
    
    Args:
        timesteps (int): Загальна кількість кроків дифузії (T)
        beta_start (float): Початкове значення beta
        beta_end (float): Кінцеве значення beta
        schedule (str): Тип розкладу ('linear' або 'cosine')
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = 'linear',
    ):
        super().__init__()
        
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        
        # Генеруємо beta розклад
        self.beta = self._get_beta_schedule()
        
        # Обчислюємо alpha = 1 - beta
        self.alpha = 1.0 - self.beta
        
        # Обчислюємо alpha_bar (кумулятивний добуток)
        # alpha_bar[t] = alpha[0] * alpha[1] * ... * alpha[t]
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Обчислюємо додаткові значення для зручності
        # sqrt(alpha_bar)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        
        # sqrt(1 - alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        
        # log(alpha_bar) для чисельної стабільності
        self.log_alpha_bar = torch.log(self.alpha_bar)
        
        # Обчислюємо значення для зворотного процесу
        # sqrt(1/alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        
        # (1 - alpha_bar) / sqrt(1 - alpha_bar)
        self.posterior_variance = (1 - self.alpha_bar[:-1]) / (1 - self.alpha_bar[1:])
        
        # Конвертуємо в тензори для GPU
        self._convert_to_tensors()
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """
        Генерує beta розклад залежно від типу schedule.
        
        Returns:
            Tensor з значеннями beta для кожного timestep
        """
        if self.schedule == 'linear':
            # Лінійний розклад (оригінальний DDPM)
            betas = torch.linspace(
                self.beta_start,
                self.beta_end,
                self.timesteps
            )
        elif self.schedule == 'cosine':
            # Косинусний розклад (покращений)
            # Формула: beta_t = f(t/T) / f(0) де f(s) = cos^2(s + s_max)/(s + 1)
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Невідомий schedule: {self.schedule}")
        
        return betas
    
    def _convert_to_tensors(self):
        """Конвертує всі значення в тензори на CPU/GPU."""
        self.beta = self.beta.to('cpu')  # Зберігаємо на CPU для індексації
        self.alpha = self.alpha.to('cpu')
        self.alpha_bar = self.alpha_bar.to('cpu')
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to('cpu')
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to('cpu')
        self.sqrt_recip_alpha_bar = self.sqrt_recip_alpha_bar.to('cpu')
        self.posterior_variance = self.posterior_variance.to('cpu')
    
    def to(self, device):
        """Переносить планувальник на пристрій."""
        self.device = device
        # Тензори залишаються на CPU для індексації, 
        # але ми повертаємо копію на потрібному пристрої
        return self
    
    def _get_value(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Отримує значення тензора для заданих timesteps.
        
        Args:
            tensor: Тензор з значеннями для всіх timesteps
            t: Tensor з індексами (batch_size,)
            
        Returns:
            Tensor з відповідними значеннями
        """
        # Конвертуємо в Python ints для індексації
        t_cpu = t.cpu().long()
        return tensor[t_cpu].to(t.device)
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Додає шум до зображення на основі формули forward process.
        
        Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_start: Оригінальні зображення (batch, channels, height, width)
            noise: Випадковий шум (такий же розмір)
            t: Timesteps для кожного зображення в батчі (batch_size,)
            
        Returns:
            Зашумлені зображення
        """
        # Отримуємо значення для timesteps
        sqrt_alpha_bar_t = self._get_value(self.sqrt_alpha_bar, t)
        sqrt_one_minus_alpha_bar_t = self._get_value(self.sqrt_one_minus_alpha_bar, t)
        
        # Змінюємо форму для broadcast
        # (batch,) -> (batch, 1, 1, 1)
        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1)
        
        # Обчислюємо зашумлене зображення
        x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Виконує один крок зворотного процесу (DDPM sampler).
        
        Це рівняння зворотного процесу:
        x_{t-1} = (x_t - (1-alpha_bar_t)/sqrt(1-alpha_bar_t) * pred_noise) / sqrt(alpha_t) + sigma_t * z
        
        де sigma_t = sqrt(beta_t * (1-alpha_bar_{t-1}) / (1-alpha_bar_t))
        
        Args:
            model_output: Передбачений шум від моделі
            timestep: Поточний timestep
            sample: Поточне зашумлене зображення
            eta: Параметр стохастичності (0 = повністю детермінований)
            
        Returns:
            Зображення на попередньому кроці
        """
        t = timestep
        
        # Отримуємо необхідні значення
        alpha_bar_t = self.alpha_bar[t].to(sample.device)
        alpha_bar_t_minus_1 = self.alpha_bar[t - 1].to(sample.device) if t > 0 else torch.tensor(1.0).to(sample.device)
        beta_t = self.beta[t].to(sample.device)
        
        # Обчислюємо коефіцієнти
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # Обчислюємо передбачене оригінальне зображення
        # x_0 = (x_t - sqrt(1-alpha_bar_t) * pred_noise) / sqrt(alpha_bar_t)
        pred_original_sample = (
            sample - sqrt_one_minus_alpha_bar_t * model_output
        ) / torch.sqrt(alpha_bar_t)
        
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        # Обчислюємо коефіцієнт для sigma
        # variance = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        pred_original_sample_variance = (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t) * beta_t
        
        # Додаємо шум (крім останнього кроку)
        if t > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(max(pred_original_sample_variance, 0)) * noise
        else:
            noise = torch.zeros_like(sample)
            variance = 0
        
        # Обчислюємо x_{t-1}
        # x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0 + sqrt(1 - alpha_bar_{t-1} - variance) * noise_prediction
        # Спрощена версія:
        pred_sample_direction = torch.sqrt(1 - alpha_bar_t_minus_1 - eta * pred_original_sample_variance) * model_output
        
        prev_sample = torch.sqrt(alpha_bar_t_minus_1) * pred_original_sample + pred_sample_direction + torch.sqrt(eta * pred_original_sample_variance) * noise
        
        return prev_sample
    
    def get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Повертає вагу втрат для кожного timestep.
        
        Деякі реалізації використовують різні ваги для різних timesteps.
        
        Args:
            t: Timesteps (batch_size,)
            
        Returns:
            Tensor з вагами
        """
        return torch.ones_like(t, dtype=torch.float32)
    
    def __repr__(self):
        return f"NoiseScheduler(timesteps={self.timesteps}, schedule='{self.schedule}')"


def create_noise_scheduler(
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    schedule: str = 'linear',
    device: str = 'cpu'
) -> NoiseScheduler:
    """
    Фабрична функція для створення NoiseScheduler.
    
    Args:
        timesteps: Кількість кроків
        beta_start: Початкове значення beta
        beta_end: Кінцеве значення beta
        schedule: Тип розкладу
        device: Пристрій
        
    Returns:
        Ініціалізований NoiseScheduler
    """
    return NoiseScheduler(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule=schedule,
    ).to(device)
