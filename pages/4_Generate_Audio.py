from audio_sample_generator import constants
from audio_sample_generator.data.model_data import ModelData
from audio_sample_generator.nn.spectrograms_generator_model import SpectrogramsGeneratorModel
from audio_sample_generator.utils.image_utils import convert_generated_data_with_model_data_to_image
from audio_sample_generator.utils.torch_utils import get_available_devices
from audio_sample_generator.utils.streamlit_utils import get_spectrogram_to_audio_settings

import torch
import torchaudio
import torchvision

import numpy as np
import streamlit as st

import random

from typing import cast

title = "Generate Audio"

st.set_page_config(
    page_icon=title,
)

st.title(title)

model_data: ModelData = st.session_state.get("model_data", None)

if model_data is None:
    st.markdown("No trained models here yet. But you can add at **Train Model**.")
else:
    with st.container(border=True):
        settings = get_spectrogram_to_audio_settings()

    with st.container(border=True):
        st.subheader("Generation Settings")

        devices = get_available_devices()

        device_name = cast(
            str,
            st.radio(
                label="Device",
                options=devices,
                horizontal=True,
                help="Please check either device type is supported on your machine.",
                index=0,
            )
        )

        seed = random.randint(constants.INT_MIN_VALUE, constants.INT_MAX_VALUE) if st.button("Randomize") else 0

        seed = cast(
            int,
            st.number_input(
                label="Seed",
                value=seed,
                step=1,
                format="%i",
                min_value=constants.INT_MIN_VALUE,
                max_value=constants.INT_MAX_VALUE,
            ),
        )

        steps = cast(
            int,
            st.number_input(
                label="Steps",
                min_value=1,
                value=20,
            ),
        )

        generate_amount = cast(
            int,
            st.number_input(
                label="Amount",
                min_value=1,
                value=1,
            ),
        )

    if st.button(
        label="Generate",
        use_container_width=True,
        type="primary",
    ):
        device = torch.device(device_name)

        torch.use_deterministic_algorithms(True)

        generator_model = SpectrogramsGeneratorModel(
            input_size=model_data.input_size,
            hidden_size=model_data.hidden_size,
            output_size=model_data.output_size,
            device=device,
        ).to(device)

        generator_model.load_state_dict(torch.load(model_data.path))

        generator_model.eval()

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

        with st.container(border=True):
            st.subheader("Generated Audio")

            placeholder_progress = st.empty()
            placeholder_preview = st.empty()

            for i in range(generate_amount):
                with torch.no_grad():
                    seed_item = seed + i

                    random.seed(seed_item)
                    np.random.seed(seed_item)
                    torch.manual_seed(seed_item)

                    data_latent = torch.randn(model_data.input_size).to(device)
                    data_generated = data_latent

                    image = None

                    for j in range(steps):
                        data_generated: torch.Tensor = generator_model(data_generated)

                        image = convert_generated_data_with_model_data_to_image(data_generated, model_data).convert("RGB")

                        placeholder_preview.image(
                            image=image.convert("RGB"),
                            output_format="PNG",
                            use_column_width="always",
                        )

                        step_count = j + 1

                        placeholder_progress.progress(
                            value=step_count / float(steps),
                            text=f"Steps: {step_count}/{steps}"
                        )

                    placeholder_preview.empty()

                    st.text("Done.")

                    mel_spectrogram = transform_image_tensor(image).squeeze(0) * 255
                    lin_spectrogram = transform_inverse_mel_scale(mel_spectrogram)
                    waveform = transform_griffin_lim(lin_spectrogram)

                    with st.container(border=True):
                        if image is not None:
                            st.image(
                                image=image.convert("RGB"),
                                output_format="PNG",
                                use_column_width="always",
                            )

                        st.audio(
                            data=waveform.numpy(),
                            sample_rate=settings.sample_rate,
                        )

