from audio_sample_generator.utils.audio_utils import load, \
                                                     trim_silence, \
                                                     trim_duration, \
                                                     pad_duration

from audio_sample_generator.utils.image_utils import convert_mel_spectrogram_to_image
from audio_sample_generator.utils.streamlit_utils import sample_data_list, common_data
from audio_sample_generator.utils.torch_utils import get_window_fn_dict

from audio_sample_generator.data.sample_data import SampleData

import streamlit as st

import torch
import torchaudio

from torchaudio.functional import resample

from typing import cast
from uuid import uuid4

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
    with st.container(border=True):
        st.subheader("Mel Spectrogram Settings")

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

        default_target_duration = st.session_state.get("default_target_duration", 1.0)

        target_duration = cast(
            float,
            st.number_input(
                label="Target Duration",
                min_value=0.01,
                max_value=10.0,
                value=default_target_duration,
                format="%.5f",
            )
        )

        default_n_mels = st.session_state.get("default_n_mels", 128)
        default_hop_length = st.session_state.get("default_hop_length", 400)
        default_custom_value_enabled_hop_length=st.session_state.get("default_custom_value_enabled_hop_length", False)

        with st.container(border=True):
            target_width = cast(
                int,
                st.number_input(
                    label="Target Width",
                    value=512,
                    step=1,
                )
            )

            target_height = cast(
                int,
                st.number_input(
                    label="Target Height",
                    value=512,
                    step=1,
                )
            )

            if st.button("Fit"):
                default_custom_value_enabled_hop_length=True

                default_n_mels = target_height
                default_hop_length = int((resample_rate * target_duration) / target_width)

                total_samples_corrected = target_width * float(default_hop_length)

                target_duration = round(total_samples_corrected / resample_rate, 3)

                st.text(f"Duration corrected to {target_duration} for fitting target width of {target_width}")

                st.session_state["default_target_duration"] = target_duration
                st.session_state["default_n_mels"] = default_n_mels
                st.session_state["default_hop_length"] = default_hop_length
                st.session_state["default_custom_value_enabled_hop_length"] = default_custom_value_enabled_hop_length

        n_fft = cast(
            int,
            st.number_input(
                label="FFT Length",
                min_value=0,
                value=4096,
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
            value=default_custom_value_enabled_hop_length,
        )

        hop_length = cast(
            int,
            st.number_input(
                label="Hop Length",
                min_value=0,
                value=default_hop_length,
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
                value=default_n_mels,
            )
        )

        window_fn_dict = get_window_fn_dict()

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

    if st.button(
        label="Extract",
        use_container_width=True,
        type="primary",
    ):
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

        with st.container(border=True):
            st.subheader("Loaded Audio")

            for input_audio_file in input_audio_files:
                try:
                    waveform_orig, sample_rate = load(input_audio_file)
                except Exception as exception:
                    with st.container(border=True):
                        st.markdown(f"Error occurred on loading of `{input_audio_file.name}`:")
                        st.error(exception)

                    continue

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

                mel_spectrogram: torch.Tensor = transform_mel_spectrogram(waveform_padded)

                sample_data = SampleData(
                    id=str(uuid4()),
                    input_audio_file_name=input_audio_file.name,
                    sample_rate=resample_rate,
                    duration=target_duration,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    f_min=f_min,
                    f_max=f_max,
                    pad=pad,
                    n_mels=n_mels,
                    window_fn_key=window_fn_key,
                    power=power,
                    normalized=normalized,
                    norm=norm,
                    mel_scale=mel_scale,
                    center=center,
                    pad_mode=pad_mode,
                    waveform=waveform_padded,
                    mel_spectrogram=mel_spectrogram,
                )

                sample_data_list.append(sample_data)

                with st.container(border=True):
                    st.text(sample_data.input_audio_file_name)

                    st.audio(
                        data=sample_data.waveform.numpy(),
                        sample_rate=sample_data.sample_rate,
                    )

                    mel_spectrogram_image = convert_mel_spectrogram_to_image(sample_data.mel_spectrogram)

                    common_data.width = max(common_data.width, mel_spectrogram_image.width)
                    common_data.height = max(common_data.height, mel_spectrogram_image.width)

                    st.image(
                        image=mel_spectrogram_image.convert("RGB"),
                        output_format="PNG",
                        use_column_width="always",
                    )

                    st.text(f"width: {mel_spectrogram_image.width}")
                    st.text(f"height: {mel_spectrogram_image.height}")

