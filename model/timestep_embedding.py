"""
Timestep Embedding (Синусоїдальне позиційне вкладення)

Цей модуль реалізує синусоїдальне позиційне вкладення (Sinusoidal Positional Embedding)
для передачі інформації про timestep (часовий крок) до U-Net моделі.

Ця техніка була вперше представлена в Transformer архітектурі (Attention is All You Need)
і успішно адаптована для diffusion моделей.

Формула:
- pos_encoding[i] = sin(pos / 10000^(2i/dim)) для парних індексів
- pos_encoding[i] = cos(pos / 10000^(2i/dim)) для непарних індексів

де:
- pos - позиція (в нашому випадку timestep)
- dim - розмірність вкладення
- i - індекс в векторі вкладення
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Синусоїдальне позиційне вкладення для timestep.
    
    Цей модуль перетворює скалярний timestep (число від 0 до T)
    у вектор фіксованої розмірності, який потім додається до 
    внутрішніх представлень U-Net.
    
    Args:
        dim (int): Розмірність вихідного вектора вкладення
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Перетворює timestep у вектор вкладення.
        
        Args:
            time: Tensor форми (batch_size,) з значеннями timestep
            
        Returns:
            Tensor форми (batch_size, dim) з позиційним вкладенням
        """
        device = time.device
        half_dim = self.dim // 2
        
        # Обчислюємо знаменник для формули: 10000^(2i/dim)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Обчислюємо кут для кожного timestep
        embeddings = time[:, None] * embeddings[None, :]
        
        # Застосовуємо sin та cos
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class GaussianFourierProjection(nn.Module):
    """
    Альтернативне вкладення на основі випадкових гаусівських проекцій.
    
    Це ще один популярний спосіб кодування timestep, який іноді
    дає кращі результати для diffusion моделей.
    
    Args:
        dim (int): Розмірність вихідного вектора
        scale (float): Масштаб для ініціалізації
    """
    
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        # Ініціалізуємо випадкові ваги з нормального розподілу
        self.register_buffer('weights', torch.randn(dim) * scale)
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Проектує вхід через гаусівські ваги та застосовує 2π.
        
        Args:
            x: Tensor форми (batch_size,) або (batch_size, 1)
            
        Returns:
            Tensor форми (batch_size, dim)
        """
        x = x.ravel()  # Перетворюємо у 1D
        x = x * 2 * torch.pi  # Масштабуємо до 2π
        x = torch.cat([torch.sin(x * self.weights), 
                       torch.cos(x * self.weights)], dim=-1)
        return x


class TimestepEmbedding(nn.Module):
    """
    Комбінований модуль для вкладення timestep з лінійним проєктором.
    
    Цей модуль об'єднує синусоїдальне вкладення з лінійним проєктором,
    який підлаштовує розмірність під архітектуру U-Net.
    
    Args:
        embed_dim (int): Розмірність синусоїдального вкладення
        hidden_dim (int): Розмірність на виході (після лінійного проєктора)
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),  # Активація Swish
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Обробляє timestep та повертає вкладення потрібної розмірності.
        
        Args:
            t: Tensor форми (batch_size,) з значеннями timestep
            
        Returns:
            Tensor форми (batch_size, hidden_dim)
        """
        return self.mlp(t)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Функція-утиліта для швидкого отримання timestep вкладення.
    
    Args:
        timesteps: Tensor форми (batch_size,) з значеннями timestep
        embedding_dim: Цільова розмірність вкладення
        
    Returns:
        Tensor форми (batch_size, embedding_dim)
    """
    half_dim = embedding_dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
    embeddings = timesteps[:, None] * embeddings[None, :]
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    return embeddings
