import torch

from platform import system
from typing import Callable, List, Dict

def get_available_devices() -> List[str]:
    devices = [
        "cpu",
        "cuda",
        "mps",
    ]

    if torch.cuda.is_available():
        devices.remove("cuda")
        devices.insert(0, "cuda")

    if torch.backends.mps.is_available() and system() == "Darwin":
        devices.remove("mps")
        devices.insert(0, "mps")

    return devices

def get_window_fn_dict() -> Dict[str, Callable]:
    return {
        "hann": torch.hann_window,
        "hamming": torch.hamming_window,
        "blackman": torch.blackman_window,
        "kaiser": torch.kaiser_window,
        "bartlett": torch.bartlett_window,
    }

