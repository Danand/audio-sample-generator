from audio_sample_generator.utils.streamlit_utils import get_spectrogram_to_audio_settings

import torchvision
import torchaudio

from PIL import Image

import streamlit as st

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
        settings = get_spectrogram_to_audio_settings()

        if st.button(
            label="Convert",
            use_container_width=True,
            type="primary",
        ):
            with st.container(border=True):
                st.subheader("Converted Audio")

                transform_inverse_mel_scale = torchaudio.transforms.InverseMelScale(
                    n_stft=settings.n_fft // 2 + 1,
                    n_mels=settings.n_mels,
                    sample_rate=settings.sample_rate,
                    f_min=settings.f_min,
                    f_max=settings.f_max,
                    norm=settings.norm,
                    mel_scale=settings.mel_scale,
                    driver=settings.driver,
                )

                transform_image_tensor = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Grayscale(),
                ])

                transform_griffin_lim = torchaudio.transforms.GriffinLim(
                    n_fft=settings.n_fft,
                    n_iter=settings.n_iter,
                    win_length=settings.win_length,
                    hop_length=settings.hop_length,
                    window_fn=settings.window_fn,
                    power=settings.power,
                    momentum=settings.momentum,
                    length=settings.length,
                    rand_init=settings.rand_init,
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
                            sample_rate=settings.sample_rate,
                        )

                        st.image(
                            image=input_image.convert("RGB"),
                            output_format="PNG",
                            use_column_width="always",
                        )
