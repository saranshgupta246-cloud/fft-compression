import numpy as np
from PIL import Image

def apply_fft_2d(channel: np.ndarray) -> np.ndarray:
    """Apply 2D FFT to a single image channel."""
    return np.fft.fft2(channel)

def compress_channel(channel, keep_fraction=0.1):
    """Zero out low-magnitude frequencies."""
    fft = apply_fft_2d(channel)
    magnitude = np.abs(fft).flatten()
    idx = int((1 - keep_fraction) * len(magnitude))
    threshold = np.sort(magnitude)[idx]
    fft_compressed = fft.copy()
    fft_compressed[np.abs(fft) < threshold] = 0
    return fft_compressed

def reconstruct_channel(fft_data: np.ndarray) -> np.ndarray:
    """Inverse FFT to reconstruct pixel data."""
    recon = np.abs(np.fft.ifft2(fft_data))
    return np.clip(recon, 0, 255).astype(np.uint8)

def compress_image_rgb(img_array, keep_fraction=0.1):
    """Full RGB image compression pipeline."""
    channels = []
    for c in range(3):
        fft_c = compress_channel(img_array[:,:,c], keep_fraction)
        channels.append(reconstruct_channel(fft_c))
    return np.stack(channels, axis=2)

def get_spectrum(channel):
    """Return log magnitude spectrum for visualization."""
    fft = np.fft.fftshift(np.fft.fft2(channel))
    return np.log(np.abs(fft) + 1)