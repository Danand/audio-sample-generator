import torch

from platform import system
from typing import List

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
