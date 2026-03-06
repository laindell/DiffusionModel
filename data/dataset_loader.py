"""
Dataset Loader для Diffusion Model

Цей модуль реалізує завантаження зображень для навчання diffusion моделі.

Підтримувані датасети:
- MNIST: рукописані цифри
- CIFAR-10: маленькі кольорові зображення
- Custom: власна папка з изображениями

Кожен датасет проходить через:
1. Завантаження
2. Ресайз до потрібного розміру
3. Нормалізацію (-1, 1) для кращої роботи з GAN/Diffusion
4. Конвертацію в тензор
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Tuple, Optional, Callable


class ImageFolderDataset(Dataset):
    """
    Кастомний Dataset для завантаження зображень з папки.
    
    Підтримує JPG, PNG та інші формати зображень.
    
    Args:
        root_dir (str): Шлях до папки з изображениями
        transform (Callable): Трансформації для застосування
        extensions (tuple): Допустимі розширення файлів
    """
    
    def __init__(
        self, 
        root_dir: str, 
        transform: Optional[Callable] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions
        
        # Рекурсивно знаходимо всі файли зображень
        self.image_paths = []
        
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(extensions):
                        self.image_paths.append(os.path.join(root, file))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"Не знайдено зображень в папці: {root_dir}")
        
        print(f"Знайдено {len(self.image_paths)} зображень")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Завантажує одне зображення за індексом.
        
        Args:
            idx: Індекс зображення
            
        Returns:
            Тензор зображення
        """
        img_path = self.image_paths[idx]
        
        # Завантажуємо зображення
        image = Image.open(img_path).convert('RGB')
        
        # Застосовуємо трансформації
        if self.transform:
            image = self.transform(image)
        
        return image


def get_transforms(image_size: int = 32, is_training: bool = True) -> transforms.Compose:
    """
    Створює стандартний набір трансформацій для зображень.
    
    Для Diffusion моделей важливо нормалізувати до [-1, 1],
    оскільки це відповідає розподілу, який модель навчена передбачати.
    
    Args:
        image_size: Цільовий розмір зображення
        is_training: Чи це режим навчання (додає аугментацію)
        
    Returns:
        Композиція трансформацій
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),  # Аугментація
            transforms.ToTensor(),  # Конвертація в [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Нормалізація до [-1, 1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def get_mnist_transforms(image_size: int = 32) -> transforms.Compose:
    """
    Спеціальні трансформації для MNIST (grayscale -> RGB).
    
    Args:
        image_size: Цільовий розмір зображення
        
    Returns:
        Композиція трансформацій
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Конвертація в RGB шляхом повторення каналу
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def load_dataset(
    dataset_name: str,
    image_size: int = 32,
    batch_size: int = 64,
    num_workers: int = 4,
    is_training: bool = True,
    data_dir: str = './data'
) -> Tuple[DataLoader, int]:
    """
    Завантажує датасет за назвою.
    
    Підтримувані назви:
    - 'mnist': рукописані цифри
    - 'cifar10': 10 класів малих зображень
    - 'folder': власна папка з зображеннями
    
    Args:
        dataset_name: Назва датасету
        image_size: Розмір зображення
        batch_size: Розмір батчу
        num_workers: Кількість потоків завантаження
        is_training: Чи це режим навчання
        data_dir: Директорія для даних
        
    Returns:
        Tuple з (DataLoader, num_channels)
    """
    
    if dataset_name.lower() == 'mnist':
        # MNIST - grayscale зображення
        transform = get_mnist_transforms(image_size)
        
        dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=is_training,
            download=True,
            transform=transform
        )
        
        num_channels = 3  # Конвертуємо в RGB
        
    elif dataset_name.lower() == 'cifar10':
        # CIFAR-10 - кольорові зображення
        transform = get_transforms(image_size, is_training)
        
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=is_training,
            download=True,
            transform=transform
        )
        
        num_channels = 3
        
    elif dataset_name.lower() == 'folder' or os.path.isdir(dataset_name):
        # Власна папка з зображеннями
        folder_path = dataset_name if os.path.isdir(dataset_name) else data_dir
        transform = get_transforms(image_size, is_training)
        
        dataset = ImageFolderDataset(
            root_dir=folder_path,
            transform=transform
        )
        
        # Визначаємо кількість каналів з першого зображення
        num_channels = 3  # За замовчуванням RGB
        
    else:
        raise ValueError(f"Невідомий датасет: {dataset_name}")
    
    # Створюємо DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader, num_channels


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Денормалізує зображення з диапазону [-1, 1] назад до [0, 1].
    
    Це корисно для візуалізації результатів.
    
    Args:
        images: Тензор зображень (batch, channels, height, width)
        
    Returns:
        Денормалізовані зображення
    """
    return (images + 1) / 2


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    """
    Зберігає сітку зображень у файл.
    
    Args:
        images: Тензор зображень (batch, channels, height, width)
        path: Шлях для збереження
        nrow: Кількість зображень в рядку
    """
    # Денормалізуємо
    images = denormalize_images(images)
    
    # Створюємо сітку
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    
    # Зберігаємо
    torchvision.utils.save_image(grid, path)


def get_dataset_info(dataset_name: str) -> dict:
    """
    Повертає інформацію про датасет.
    
    Args:
        dataset_name: Назва датасету
        
    Returns:
        Словник з інформацією
    """
    info = {
        'mnist': {
            'num_classes': 10,
            'image_size': 28,
            'channels': 1,
            'description': 'Рукописані цифри 0-9'
        },
        'cifar10': {
            'num_classes': 10,
            'image_size': 32,
            'channels': 3,
            'description': '10 класів обєктів (літаки, машини, птахи...)'
        }
    }
    
    return info.get(dataset_name.lower(), {'description': 'Кастомний датасет'})
