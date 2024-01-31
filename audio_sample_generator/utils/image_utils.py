from audio_sample_generator.data.model_data import ModelData

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

def convert_generated_data_with_model_data_to_image(
    data_generated: torch.Tensor,
    model_data: ModelData,
) -> Image.Image:
    return convert_generated_data_to_image(
        data_generated=data_generated,
        normalization_std=model_data.normalization_std,
        normalization_mean=model_data.normalization_mean,
        output_height=model_data.output_height,
        output_width=model_data.output_width,
    )

def convert_generated_data_to_image(
    data_generated: torch.Tensor,
    normalization_std: torch.Tensor,
    normalization_mean: torch.Tensor,
    output_height: int,
    output_width: int,
) -> Image.Image:
    data_generated_denormalized = (data_generated * normalization_std) + normalization_mean
    data_generated_descaled = data_generated_denormalized * 255
    data_image = data_generated_descaled.view(output_height, output_width)

    image_array = data_image.detach() \
                            .cpu() \
                            .numpy() \
                            .astype(np.uint8)

    return Image.fromarray(
        obj=image_array,
        mode="L",
    )


