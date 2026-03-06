"""
Конфігурація проєкту Diffusion Model

Цей файл містить усі гіперпараметри та налаштування для:
- Параметрів навчання
- Архітектури моделі
- Дифузійного процесу
- Датасету
"""

import os
import torch

# ============================================================================
# Основні налаштування
# ============================================================================

# Розмір зображення (для швидкого навчання рекомендується 32x32 або 64x64)
IMAGE_SIZE = 64

# Розмір батчу
BATCH_SIZE = 64

# Кількість епох навчання
EPOCHS = 50

# Learning rate для оптимізатора
LEARNING_RATE = 1e-4

# Вага L2 регуляризації (weight decay)
WEIGHT_DECAY = 0.0

# Крок збереження чекпоінтів (кожні N епох)
CHECKPOINT_EVERY = 5

# Крок візуалізації (кожні N епох)
VISUALIZE_EVERY = 5

# ============================================================================
# Налаштування Diffusion процесу (DDPM)
# ============================================================================

# Кількість timesteps (кроків дифузії)
# Типові значення: 1000, 500, 100
TIMESTEPS = 1000

# Вибір бета-розкладу: 'linear' або 'cosine'
BETA_SCHEDULE = 'linear'

# Для linear schedule - початкове та кінцеве значення beta
BETA_START = 1e-4
BETA_END = 0.02

# Розмір вкладення для timestep (dim для SinusoidalEmbedding)
TIME_EMBED_DIM = 128

# ============================================================================
# Архітектура U-Net
# ============================================================================

# Кількість каналів на першому рівні
BASE_CHANNELS = 64

# Кількість блоків на кожному рівні
CHANNELS_MULTIPLIER = [1, 2, 4, 8]

# Кількість residual блоків на рівень
NUM_RES_BLOCKS = 2

# Використовувати attention механізм
ATTENTION_RESOLUTIONS = [8, 16]

# Використовувати dropout
DROPOUT = 0.1

# ============================================================================
# Налаштування датасету
# ============================================================================

# Шлях до датасету (папка з изображениями)
# Можна використовувати: 'mnist', 'cifar10', або шлях до папки
DATASET_PATH = 'dataset/jpg'

# Кількість робочих потоків для завантаження даних
NUM_WORKERS = 4

# ============================================================================
# Обчислювальні налаштуння
# ============================================================================

# Використовувати GPU якщо доступно
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Автоматичний вибір Mixed Precision залежно від покоління відеокарти
if DEVICE == 'cuda' and torch.cuda.is_bf16_supported():
    # Для RTX 30xx, 40xx і новіших
    USE_MIXED_PRECISION = True
    AUTOCAST_DTYPE = torch.bfloat16
    print("-> Відеокарта підтримує bfloat16: Mixed Precision УВІМКНЕНО")
else:
    # Для RTX 20xx, GTX 10xx, 16xx (або якщо використовується CPU)
    USE_MIXED_PRECISION = False
    AUTOCAST_DTYPE = torch.float32
    print("-> Відеокарта НЕ підтримує bfloat16: Mixed Precision ВИМКНЕНО (використовується float32 для стабільності)")

# Крок градієнту (gradient clipping)
GRADIENT_CLIP = 1.0

# ============================================================================
# Шляхи
# ============================================================================

# Директорія для збереження результатів
OUTPUT_DIR = 'outputs'

# Директорія для збереження чекпоінтів
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

# Директорія для збереження зображень
SAMPLES_DIR = os.path.join(OUTPUT_DIR, 'samples')

# ============================================================================
# Допоміжні функції
# ============================================================================

def get_config():
    """
    Повертає словник з усіма налаштуваннями.
    Корисно для передачі в інші модулі.
    """
    return {
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'checkpoint_every': CHECKPOINT_EVERY,
        'visualize_every': VISUALIZE_EVERY,
        'timesteps': TIMESTEPS,
        'beta_schedule': BETA_SCHEDULE,
        'beta_start': BETA_START,
        'beta_end': BETA_END,
        'time_embed_dim': TIME_EMBED_DIM,
        'base_channels': BASE_CHANNELS,
        'channels_multiplier': CHANNELS_MULTIPLIER,
        'num_res_blocks': NUM_RES_BLOCKS,
        'attention_resolutions': ATTENTION_RESOLUTIONS,
        'dropout': DROPOUT,
        'dataset_path': DATASET_PATH,
        'num_workers': NUM_WORKERS,
        'device': DEVICE,
        'use_mixed_precision': USE_MIXED_PRECISION,
        'autocast_dtype': str(AUTOCAST_DTYPE),
        'gradient_clip': GRADIENT_CLIP,
        'output_dir': OUTPUT_DIR,
        'checkpoint_dir': CHECKPOINT_DIR,
        'samples_dir': SAMPLES_DIR,
    }


def create_directories():
    """
    Створює необхідні директорії для збереження результатів.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

