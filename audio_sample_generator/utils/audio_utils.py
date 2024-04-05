import torch

# HACK: Fix linting failure on `torchaudio.load()`
from torchaudio._backend.utils import get_load_func, get_available_backends

import soundfile

import os

from typing import BinaryIO, Tuple, Union

def load_with_found_backend(
    file: Union[BinaryIO, str, os.PathLike],
) -> Tuple[torch.Tensor, int]:
    load_func = get_load_func()
    return load_func(file)

def load_with_fallback_to_soundfile(
    file: Union[BinaryIO, str, os.PathLike],
) -> Tuple[torch.Tensor, int]:
    waveform_ndarray, sample_rate = soundfile.read(
        file=file,
        dtype="float32",
        always_2d=True,
    )

    waveform_tensor = torch.from_numpy(waveform_ndarray) \
                           .permute(1, 0)

    return waveform_tensor, sample_rate

available_backends = get_available_backends()

if len(available_backends) > 0:
    load = load_with_found_backend
else:
    load = load_with_fallback_to_soundfile

def trim_silence(
    waveform: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    channel = waveform[0]

    first_index = 0

    for index, value in enumerate(channel):
        if value <= threshold:
            first_index = index
        else:
            break

    waveform_trimmed = torch.unsqueeze(channel[first_index:], 0)

    return waveform_trimmed

def trim_duration(
    waveform: torch.Tensor,
    duration: float,
    sample_rate: int,
) -> torch.Tensor:
    last_index = int(sample_rate * duration)

    sample_length = waveform.size(1)

    if sample_length <= last_index:
        return waveform

    channel = waveform[0]

    waveform_trimmed = torch.unsqueeze(channel[:last_index], 0)

    return waveform_trimmed

def pad_duration(
    waveform: torch.Tensor,
    duration: float,
    sample_rate: int,
) -> torch.Tensor:
    last_index = int(sample_rate * duration)

    sample_length = waveform.size(1)

    pad_size = last_index - sample_length

    if pad_size <= 0:
        return waveform

    waveform_padded = torch.nn.functional.pad(
        input=waveform,
        pad=(0, pad_size),
        value=0,
    )

    return waveform_padded

