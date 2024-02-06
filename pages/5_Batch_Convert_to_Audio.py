from audio_sample_generator.utils.torch_utils import get_window_fn_dict

import torchvision
import torchaudio

from PIL import Image

import streamlit as st

from typing import cast

title = "Batch Convert Spectrograms To Audio"

st.set_page_config(
    page_icon=title,
)

st.title(title)

input_image_files = st.file_uploader(
    label="Input Audio Files",
    accept_multiple_files=True,
    type=[
        "png",
    ],
)

if input_image_files is not None \
    and len(input_image_files) > 0:
    with st.container(border=True):
        st.subheader("Mel Spectrogram Settings")

        sample_rate = cast(
            int,
            st.selectbox(
                label="Sample rate",
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

        f_min = cast(
            float,
            st.number_input(
                label="Minimum Frequency",
                min_value=0.0,
                value=0.0,
            )
        )

        custom_value_enabled_f_max = st.checkbox(
            label="Custom Maximum Frequency",
            value=False,
        )

        f_max = cast(
            float,
            st.number_input(
                label="Maximum Frequency",
                min_value=0.0,
                value=float(sample_rate // 2),
                step=1.0,
                disabled=not custom_value_enabled_f_max,
            )
        )

        if not custom_value_enabled_f_max:
            f_max = None

        norm = st.selectbox(
            label="Area Normalization",
            options=[
                None,
                "slaney",
            ],
            index=0,
        )

        n_fft = cast(
            int,
            st.number_input(
                label="FFT Length",
                min_value=0,
                value=400,
            )
        )

        n_iter = cast(
            int,
            st.number_input(
                label="Phase Recovery Iterations",
                min_value=1,
                value=32,
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

        hop_length = cast(
            int,
            st.number_input(
                label="Hop Length",
                min_value=0,
                value=400,
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

        window_fn_dict = get_window_fn_dict()

        window_fn_key = cast(
            str,
            st.selectbox(
                label="Window Function",
                options=window_fn_dict.keys(),
                index=0,
            )
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

        driver = cast(
            str,
            st.selectbox(
                label="Driver",
                options=[
                    "gels",
                    "gelsy",
                    "gelsd",
                    "gelss",
                ],
                index=1,
            )
        )

        window_fn = window_fn_dict[window_fn_key]

        power = cast(
            float,
            st.number_input(
                label="Power",
                min_value=0.0,
                value=1.0,
            )
        )

        momentum = cast(
            float,
            st.number_input(
                label="Momentum",
                min_value=0.0,
                value=0.99,
                help="The momentum parameter for fast Griffin-Lim. Setting this to 0 recovers the original Griffin-Lim method. Values near 1 can lead to faster convergence, but above 1 may not converge.",
            )
        )

        custom_value_enabled_length = st.checkbox(
            label="Custom Length",
            value=False,
        )

        length = cast(
            int,
            st.number_input(
                label="Length",
                min_value=0,
                value=int(sample_rate / hop_length),
                disabled=not custom_value_enabled_length,
            )
        )

        rand_init = st.checkbox(
            label="Random Phase",
            value=True,
        )

        if not custom_value_enabled_length:
            length = None

        if st.button(
            label="Convert",
            use_container_width=True,
            type="primary",
        ):
            with st.container(border=True):
                st.subheader("Converted Audio")

                transform_inverse_mel_scale = torchaudio.transforms.InverseMelScale(
                    n_stft=n_fft // 2 + 1,
                    n_mels=n_mels,
                    sample_rate=sample_rate,
                    f_min=f_min,
                    f_max=f_max,
                    norm=norm,
                    mel_scale=mel_scale,
                    driver=driver,
                )

                transform_image_tensor = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Grayscale(),
                ])

                transform_griffin_lim = torchaudio.transforms.GriffinLim(
                    n_fft=n_fft,
                    n_iter=n_iter,
                    win_length=win_length,
                    hop_length=hop_length,
                    window_fn=window_fn,
                    power=power,
                    momentum=momentum,
                    length=length,
                    rand_init=rand_init,
                )

                for input_image_file in input_image_files:
                    input_image = Image.open(input_image_file)

                    mel_spectrogram = transform_image_tensor(input_image).squeeze(0) * 255
                    lin_spectrogram = transform_inverse_mel_scale(mel_spectrogram)
                    waveform = transform_griffin_lim(lin_spectrogram)

                    with st.container(border=True):
                        st.text(input_image_file.name)

                        st.audio(
                            data=waveform.numpy(),
                            sample_rate=sample_rate,
                        )

                        st.image(
                            image=input_image.convert("RGB"),
                            output_format="PNG",
                            use_column_width="always",
                        )
