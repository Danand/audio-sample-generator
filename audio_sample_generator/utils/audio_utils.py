import torch

from torchaudio._backend.utils import get_load_func

# HACK: Fix linting failure on `torchaudio.load()`
load = get_load_func()

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

