import torch

import numpy as np

from PIL import Image

def convert_mel_spectrogram_to_image(mel_spectrogram: torch.Tensor) -> Image.Image:
    mel_spectrogram_image_data = mel_spectrogram.squeeze(0) \
                                                .numpy() \
                                                .astype(np.float32)

    mel_spectrogram_image = Image.fromarray(
        obj=mel_spectrogram_image_data,
        mode="F",
    ).convert("L")

    return mel_spectrogram_image

