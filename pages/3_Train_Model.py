from audio_sample_generator import constants
from audio_sample_generator.utils.image_utils import convert_generated_data_to_image
from audio_sample_generator.utils.streamlit_utils import sample_data_list, common_data
from audio_sample_generator.utils.torch_utils import get_available_devices
from audio_sample_generator.nn.spectrograms_generator_model import SpectrogramsGeneratorModel
from audio_sample_generator.data.model_data import ModelData
from audio_sample_generator.constants import DATASET_ROOT_DIR

import torch

from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from PIL import Image

import numpy as np
import streamlit as st

import random

from typing import cast
from os import makedirs
from typing import cast
from shutil import rmtree

def load_image(path: str) -> Image.Image:
    return Image.open(path)

title = "Train Model"

st.set_page_config(
    page_icon=title,
)

st.title(title)

if (len(sample_data_list) == 0):
    st.markdown("No dataset items here yet. But you can add at **Prepare Dataset**.")
else:
    with st.container(border=True):
        st.subheader("Train Spectrograms Model")

        output_height = common_data.height
        output_width = common_data.width
        input_size = output_width * output_height

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

        st.text_input(
            label="Input Size",
            value=input_size,
            disabled=True,
        )

        hidden_size = cast(
            int,
            st.number_input(
                label="Hidden Size",
                min_value=1,
                value=int(input_size / 2),
                step=1,
            ),
        )

        learning_rate = cast(
            float,
            st.number_input(
                label="Learning Rate",
                min_value=0.0000001,
                value=0.0001,
                step=0.0000001,
                format="%.7f",
            ),
        )

        num_epochs = cast(
            int,
            st.number_input(
                label="Epochs",
                min_value=1,
                value=10,
            ),
        )

        batch_size = cast(
            int,
            st.number_input(
                label="Batch Size",
                min_value=1,
                value=1,
            ),
        )

        is_enabled_preview = st.checkbox(
            label="Preview Enabled",
            value=True,
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

        output_size = input_size

        device = torch.device(device_name)

        normalization_mean = torch.tensor([
            0.5,
        ]).to(device)

        normalization_std = torch.tensor([
            0.5,
        ]).to(device)

        if st.button(
            label="Train",
            use_container_width=True,
            type="primary",
        ):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)

            generator_model = SpectrogramsGeneratorModel(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                device=device,
            ).to(device)

            criterion = nn.MSELoss().to(device)

            optimizer = optim.AdamW(
                params=generator_model.parameters(),
                lr=learning_rate,
            )

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalization_mean, normalization_std),
            ])

            train_dataset = datasets.ImageFolder(
                root=DATASET_ROOT_DIR,
                transform=transform,
                loader=load_image,
            )

            random_generator = torch.Generator()
            random_generator.manual_seed(seed)

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=random_generator,
            )

            placeholder_progress = st.empty()

            column_epoch, column_step, column_loss = st.columns(3)

            placeholder_epoch = column_epoch.empty()
            placeholder_step = column_step.empty()
            placeholder_loss = column_loss.empty()

            placeholder_preview = st.empty()

            step_count = 0

            for epoch in range(num_epochs):
                for batch_idx, (data, _) in enumerate(train_loader):
                    data_flatten = data.view(-1).to(device)

                    outputs = generator_model(data_flatten)
                    loss = criterion(outputs, data_flatten)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step_count += 1

                    placeholder_progress.progress(
                        value=step_count / float(len(train_loader) * num_epochs),
                    )

                    placeholder_epoch.text(f"Epoch: {epoch + 1} / {num_epochs}")
                    placeholder_step.text(f"Step: {step_count}/{(len(train_loader) * num_epochs)}")
                    placeholder_loss.text(f"Loss: {loss.item():.4f}")

                    if is_enabled_preview:
                        image_preview = convert_generated_data_to_image(
                            data_generated=outputs,
                            normalization_std=normalization_std,
                            normalization_mean=normalization_mean,
                            output_height=output_height,
                            output_width=output_width,
                        )

                        placeholder_preview.image(
                            image=image_preview.convert("RGB"),
                            output_format="PNG",
                            use_column_width="always",
                        )

            model_output_dir="./temp/models"

            rmtree(
                path=model_output_dir,
                ignore_errors=True,
            )

            makedirs(
                name=model_output_dir,
                exist_ok=True,
            )

            model_output_path=f"{model_output_dir}/spectrograms.pth"

            torch.save(
                obj=generator_model.state_dict(),
                f=model_output_path,
            )

            model_data = ModelData(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                path=model_output_path,
                output_height=output_height,
                output_width=output_width,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
            )

            st.session_state["model_data"] = model_data

            st.text("Done.")

            st.code(
                body=model_output_path,
                language="bash",
            )

