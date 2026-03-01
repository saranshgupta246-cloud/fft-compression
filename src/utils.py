import numpy as np
from PIL import Image
import os

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float64)

def save_image(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)

def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))

def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Scale spectrum to 0-255 for display."""
    norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
    return (norm * 255).astype(np.uint8)