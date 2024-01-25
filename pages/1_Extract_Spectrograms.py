import streamlit as st

import numpy as np

import torch
import torchaudio

from torchaudio._backend.utils import get_load_func

# HACK: Fix linting failure on `torchaudio.load()`
load = get_load_func()

from torchaudio.functional import resample

from PIL import Image

from typing import cast

def trim_silence(waveform: torch.Tensor, threshold: float) -> torch.Tensor:
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

title = "Extract Spectrograms"

st.set_page_config(
    page_icon=title,
)

st.title(title)

input_audio_files = st.file_uploader(
    label="Input Audio Files",
    accept_multiple_files=True,
    type=[
        "wav",
    ],
)

if input_audio_files is not None \
   and len(input_audio_files) > 0:
    resample_rate = cast(
        int,
        st.selectbox(
            label="Resample rate",
            options=[
                8000,
                11025,
                16000,
                22050,
                44100,
                48000,
            ],
            index=3,
        )
    )

    silence_threshold = cast(
        float,
        st.number_input(
            label="Silence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
        )
    )

    target_duration = cast(
        float,
        st.number_input(
            label="Target Duration",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
        )
    )

    n_fft = cast(
        int,
        st.number_input(
            label="FFT Length",
            min_value=0,
            value=400,
        )
    )

    custom_value_enabled_win_length = st.checkbox(
        label="Custom Window Length",
        value=False,
    )

    win_length = cast(
        int,
        st.number_input(
            label="Window Length",
            min_value=0,
            value=400,
            disabled=not custom_value_enabled_win_length,
        )
    )

    if not custom_value_enabled_win_length:
        win_length = None

    custom_value_enabled_hop_length = st.checkbox(
        label="Custom Hop Length",
        value=False,
    )

    hop_length = cast(
        int,
        st.number_input(
            label="Hop Length",
            min_value=0,
            value=400,
            disabled=not custom_value_enabled_hop_length,
        )
    )

    if not custom_value_enabled_hop_length:
        hop_length = None

    f_min = cast(
        float,
        st.number_input(
            label="Frequency Minimum",
            min_value=0.0,
            value=0.0,
        )
    )

    custom_value_enabled_f_max = st.checkbox(
        label="Custom Frequency Maximum",
        value=False,
    )

    f_max = cast(
        float,
        st.number_input(
            label="Frequency Maximum",
            min_value=0.0,
            max_value=22050.0,
            value=0.0,
            disabled=not custom_value_enabled_f_max,
        )
    )

    if not custom_value_enabled_f_max:
        f_max = None

    pad = cast(
        int,
        st.number_input(
            label="Pad Size",
            min_value=0,
            value=0,
        )
    )

    n_mels = cast(
        int,
        st.number_input(
            label="Mel filter banks number",
            min_value=0,
            value=128,
        )
    )

    window_fn_dict = {
        "hann": torch.hann_window,
        "hamming": torch.hamming_window,
        "blackman": torch.blackman_window,
        "kaiser": torch.kaiser_window,
        "bartlett": torch.bartlett_window,
    }

    window_fn_key = cast(
        str,
        st.selectbox(
            label="Window Function",
            options=window_fn_dict.keys(),
            index=0,
        )
    )

    window_fn = window_fn_dict[window_fn_key]

    power = cast(
        float,
        st.number_input(
            label="Power",
            min_value=0.0,
            value=2.0,
        )
    )

    normalized = st.checkbox(
        label="Normalize by Magnitude",
        value=False,
    )

    norm = st.selectbox(
        label="Area Normalization",
        options=[
            None,
            "slaney",
        ],
        index=0,
    )

    mel_scale = cast(
        str,
        st.selectbox(
            label="Mel Scale",
            options=[
                "htk",
                "slaney",
            ],
            index=0,
        )
    )

    center = st.checkbox(
        label="Center",
        value=True,
    )

    pad_mode = cast(
        str,
        st.selectbox(
            label="Pad Mode",
            options=[
                "reflect",
            ],
            index=0,
        )
    )

    if st.button("Extract"):
        for input_audio_file in input_audio_files:
            waveform_orig, sample_rate = load(input_audio_file)

            waveform_resampled = resample(
                waveform=waveform_orig,
                orig_freq=sample_rate,
                new_freq=resample_rate,
            )

            waveform_mono = torch.mean(
                input=waveform_resampled,
                dim=0,
                keepdim=True,
            )

            waveform_trimmed_silence = trim_silence(
                waveform=waveform_mono,
                threshold=silence_threshold,
            )

            waveform_trimmed_duration = trim_duration(
                waveform=waveform_trimmed_silence,
                duration=target_duration,
                sample_rate=resample_rate,
            )

            waveform_padded = pad_duration(
                waveform=waveform_trimmed_duration,
                duration=target_duration,
                sample_rate=resample_rate,
            )

            st.audio(
                data=waveform_padded.numpy(),
                sample_rate=resample_rate,
            )

            transform_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=resample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                pad=pad,
                n_mels=n_mels,
                window_fn=window_fn,
                power=power,
                normalized=normalized,
                norm=norm,
                mel_scale=mel_scale,
                center=center,
                pad_mode=pad_mode,
            )

            mel_spectrogram: torch.Tensor = transform_mel_spectrogram(waveform_padded)

            mel_spectrogram_image_data = (mel_spectrogram).squeeze(0) \
                                                          .numpy() \
                                                          .astype(np.float32)

            mel_spectrogram_image = Image.fromarray(
                obj=mel_spectrogram_image_data,
                mode="F",
            ).convert("RGB")

            st.image(
                image=mel_spectrogram_image,
                output_format="PNG",
            )
