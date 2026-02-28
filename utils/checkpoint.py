"""
Checkpoint Management

Цей модуль відповідає за збереження та завантаження ваг моделі,
а також стану оптимізатора для відновлення навчання.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import json


class CheckpointManager:
    """
    Менеджер для збереження та завантаження чекпоінтів.
    
    Зберігає:
    - Ваги моделі
    - Стан оптимізатора
    - Значення learning rate
    - Поточну епоху
    - Історію loss
    - Конфігурацію
    """
    
    def __init__(self, checkpoint_dir: str = 'outputs/checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        config: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """
        Зберігає чекпоінт.
        
        Args:
            model: Модель для збереження
            optimizer: Оптимізатор
            epoch: Поточна епоха
            loss: Поточний loss
            config: Конфігурація
            filename: Ім'я файлу (якщо None - генерується автоматично)
            
        Returns:
            Шлях до збереженого файлу
        """
        if filename is None:
            filename = f'checkpoint_epoch_{epoch:04d}.pt'
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config,
        }
        
        torch.save(checkpoint, filepath)
        
        # Також зберігаємо latest чекпоінт
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        print(f"Чекпоінт збережено: {filepath}")
        
        return filepath
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: str = 'latest.pt',
        device: str = 'cpu',
    ) -> Dict[str, Any]:
        """
        Завантажує чекпоінт.
        
        Args:
            model: Модель для завантаження ваг
            optimizer: Оптимізатор (опціонально)
            filename: Ім'я файлу
            device: Пристрій для завантаження
            
        Returns:
            Словник з даними чекпоінту
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Чекпоінт не знайдено: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Завантажуємо ваги моделі
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Завантажуємо стан оптимізатора
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Чекпоінт завантажено: {filepath}")
        
        return checkpoint
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """
        Повертає шлях до останнього чекпоінту.
        
        Returns:
            Шлях до файлу або None якщо немає чекпоінтів
        """
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        if os.path.exists(latest_path):
            return latest_path
        return None
    
    def list_checkpoints(self) -> list:
        """
        Повертає список усіх чекпоінтів.
        
        Returns:
            Список імен файлів чекпоінтів
        """
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.endswith('.pt')
        ]
        
        return sorted(checkpoints)
    
    def delete_old_checkpoints(self, keep_last: int = 5):
        """
        Видаляє старі чекпоінти, залишаючи тільки останні N.
        
        Args:
            keep_last: Кількість чекпоінтів для збереження
        """
        checkpoints = self.list_checkpoints()
        
        # Видаляємо latest з списку
        checkpoints = [c for c in checkpoints if c != 'latest.pt']
        
        # Сортуємо за часом створення
        checkpoints_with_time = []
        for ckpt in checkpoints:
            filepath = os.path.join(self.checkpoint_dir, ckpt)
            mtime = os.path.getmtime(filepath)
            checkpoints_with_time.append((mtime, ckpt))
        
        checkpoints_with_time.sort()
        
        # Видаляємо старі
        if len(checkpoints_with_time) > keep_last:
            for _, filename in checkpoints_with_time[:-keep_last]:
                filepath = os.path.join(self.checkpoint_dir, filename)
                os.remove(filepath)
                print(f"Видалено старий чекпоінт: {filename}")


def save_model(model: nn.Module, path: str):
    """
    Зберігає лише ваги моделі (без оптимізатора).
    
    Args:
        model: Модель
        path: Шлях для збереження
    """
    torch.save(model.state_dict(), path)
    print(f"Модель збережено: {path}")


def load_model(model: nn.Module, path: str, device: str = 'cpu') -> nn.Module:
    """
    Завантажує ваги моделі.
    
    Args:
        model: Модель (архітектура має відповідати)
        path: Шлях до файлу
        device: Пристрій
        
    Returns:
        Модель з завантаженими вагами
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Модель завантажено: {path}")
    return model
