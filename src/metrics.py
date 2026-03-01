import numpy as np

def mse(original, compressed):
    """Mean Squared Error — lower is better."""
    return np.mean((original.astype(float) - compressed.astype(float))**2)

def psnr(original, compressed):
    """Peak Signal-to-Noise Ratio — higher is better. >30 dB = good."""
    error = mse(original, compressed)
    if error == 0: return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(error))

def compression_ratio(keep_fraction):
    """How much data was removed."""
    return (1 - keep_fraction) * 100

def full_report(original, compressed, keep_fraction):
    return {
        "MSE": round(mse(original, compressed), 2),
        "PSNR (dB)": round(psnr(original, compressed), 2),
        "Compressed (%)": round(compression_ratio(keep_fraction), 1),
        "Data Kept (%)": round(keep_fraction * 100, 1)
    }