"""
Блоки для U-Net архітектури

Цей модуль містить основні будівельні блоки для U-Net:
- ResidualBlock (залишковий блок з timestep conditioning)
- AttentionBlock (блок уваги для захоплення глобальних залежностей)
- Downsample/Upsample блоки для енкодера/декодера
- Normalization та activation шари
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Залишковий блок (Residual Block) з timestep conditioning.
    
    Цей блок є основою U-Net в diffusion моделях. Він:
    1. Застосовує Group Normalization
    2. Додає timestep вкладення через linear projection
    3. Використовує два Conv3x3 шари з Swish активацією
    
    Args:
        in_channels (int): Кількість вхідних каналів
        out_channels (int): Кількість вихідних каналів
        time_embed_dim (int): Розмірність timestep вкладення
        dropout (float): Ймовірність dropout
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, dropout: float = 0.0):
        super().__init__()
        
        # Перший конволюційний блок з нормалізацією
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # norm1 applies before conv1, so it must normalize input channels.
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()  # Swish активація
        
        # Проєктор для timestep вкладення
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, out_channels),
            nn.SiLU(),
        )
        
        # Другий конволюційний блок
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        
        # Dropout для регуляризації
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection: якщо розміри відрізняються, проєктуємо
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід через блок.
        
        Args:
            x: Вхідний тензор (batch, channels, height, width)
            time_embed: Timestep вкладення (batch, time_embed_dim)
            
        Returns:
            Вихідний тензор з skip connection
        """
        # Зберігаємо оригінал для skip connection
        h = self.skip(x)
        
        # Перший блок: norm -> conv -> act
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        
        # Додаємо timestep conditioning
        # time_embed: (batch, time_embed_dim) -> (batch, out_channels, 1, 1)
        time_embed = self.time_mlp(time_embed)
        x = x + time_embed[:, :, None, None]
        
        # Другий блок: norm -> conv -> act -> dropout
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = self.dropout(x)
        
        # Додаємо skip connection
        return h + x


class AttentionBlock(nn.Module):
    """
    Блок Self-Attention для захоплення глобальних залежностей.
    
    Цей блок використовує multi-head self-attention механізм,
    який дозволяє моделі фокусуватися на важливих частинах зображення.
    
    Args:
        channels (int): Кількість каналів
        num_heads (int): Кількість голів уваги
    """
    
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Нормалізація
        self.norm = nn.GroupNorm(32, channels)
        
        # Q, K, V проєкції
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        
        # Вихідна проєкція
        self.proj = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід через блок уваги.
        
        Args:
            x: Вхідний тензор (batch, channels, height, width)
            
        Returns:
            Вихідний тензор з доданою увагою
        """
        
        batch, channels, height, width = x.shape
        
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(batch, channels, height * width)
        
        qkv = self.qkv(x_flat)
        q, k, v = qkv.split(self.channels, dim=1)
        
        # Змінюємо форму і транспонуємо для F.scaled_dot_product_attention
        q = q.reshape(batch, self.num_heads, self.head_dim, height * width).transpose(-2, -1)
        k = k.reshape(batch, self.num_heads, self.head_dim, height * width).transpose(-2, -1)
        v = v.reshape(batch, self.num_heads, self.head_dim, height * width).transpose(-2, -1)
        
        # Використовуємо вбудований оптимізований Flash Attention
        out = F.scaled_dot_product_attention(q, k, v)
        
        # Повертаємо початкову форму
        out = out.transpose(-2, -1).reshape(batch, channels, height * width)
        
        out = self.proj(out)
        out = out.reshape(batch, channels, height, width)
        return x + out


class Downsample(nn.Module):
    """
    Downsampling блок для енкодера.
    
    Зменшує просторову розмірність в 2 рази.
    
    Args:
        channels (int): Кількість каналів
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling блок для декодера.
    
    Збільшує просторову розмірність в 2 рази.
    
    Args:
        channels (int): Кількість каналів
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Використовуємо інтерполяцію + conv для кращої якості
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class ConvBlock(nn.Module):
    """
    Простий конволюційний блок з нормалізацією та активацією.
    
    Args:
        in_channels (int): Вхідні канали
        out_channels (int): Вихідні канали
        kernel_size (int): Розмір ядра (3 за замовчуванням)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def make_attention_resolution(resolutions: list, num_resolutions: int) -> list:
    """
    Допоміжна функція для визначення, на яких розділеннях застосовувати attention.
    
    Args:
        resolutions: Список можливих розділень
        num_resolutions: Кількість рівнів у U-Net
        
    Returns:
        Список розділень для attention
    """
    # Для кожного рівня обчислюємо, чи потрібен attention
    attn_resolutions = []
    for i in range(num_resolutions):
        resolution = resolutions[i]
        # Attention на менших роздільних здатностях (8, 16)
        if resolution in [8, 16]:
            attn_resolutions.append(resolution)
    return attn_resolutions
