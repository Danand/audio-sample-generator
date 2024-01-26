from dataclasses import dataclass

from torch import Tensor

@dataclass
class SampleData:
    id: str
    input_audio_file_name: str
    sample_rate: int
    duration: float
    n_fft: int
    win_length: int | None
    hop_length: int | None
    f_min: float
    f_max: float | None
    pad: int
    n_mels: int
    window_fn_key: str
    power: float
    normalized: bool
    norm: str | None
    mel_scale: str
    center: bool
    pad_mode: str
    waveform: Tensor
    mel_spectrogram: Tensor
    is_enabled_for_training = True
    caption: str | None = None
    subject: str | None = None
    weight: float = 1.0

