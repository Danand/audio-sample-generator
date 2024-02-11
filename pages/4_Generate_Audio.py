from audio_sample_generator import constants
from audio_sample_generator.data.model_data import ModelData
from audio_sample_generator.nn.spectrograms_generator_model import SpectrogramsGeneratorModel
from audio_sample_generator.utils.image_utils import convert_generated_data_with_model_data_to_image
from audio_sample_generator.utils.torch_utils import get_available_devices

import torch

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
        st.subheader("Generate Audio with Trained Model")

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

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)

            generator_model = SpectrogramsGeneratorModel(
                input_size=model_data.input_size,
                hidden_size=model_data.hidden_size,
                output_size=model_data.output_size,
                device=device,
            ).to(device)

            generator_model.load_state_dict(torch.load(model_data.path))

            generator_model.eval()

            placeholder_progress = st.empty()
            placeholder_preview = st.empty()

            for i in range(generate_amount):
                with torch.no_grad():
                    data_latent = torch.randn(model_data.input_size).to(device)
                    data_generated = data_latent

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

                        # TODO: Convert to audio.

