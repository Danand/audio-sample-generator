from audio_sample_generator.utils.image_utils import convert_mel_spectrogram_to_image
from audio_sample_generator.utils.streamlit_utils import sample_data_list
from audio_sample_generator.nn.spectrograms import SpectrogramsModule

import torch

from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import streamlit as st

from os import makedirs
from typing import cast
from shutil import rmtree

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

        dataset_root_dir = "./temp/dataset"

        input_size = 0

        for sample_data in sample_data_list:
            mel_spectrogram_image = convert_mel_spectrogram_to_image(sample_data.mel_spectrogram)

            class_name = "spectrograms" if sample_data.subject is None else sample_data.subject

            class_dir = f"{dataset_root_dir}/{class_name}"

            rmtree(
                path=class_dir,
                ignore_errors=True,
            )

            makedirs(
                name=class_dir,
                exist_ok=True,
            )

            mel_spectrogram_image_path = f"{class_dir}/{sample_data.id}.png"
            mel_spectrogram_image.save(mel_spectrogram_image_path)

            st.text(f"Saved image for training: '{mel_spectrogram_image_path}'")

            input_size = max(
                input_size,
                mel_spectrogram_image.height * mel_spectrogram_image.width,
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
                value=0.001,
                step=0.0000001,
                format="%.7f",
            ),
        )

        num_epochs = cast(
            int,
            st.number_input(
                label="Epochs",
                min_value=1,
                value=1,
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

        output_size = input_size

        if st.button(
            label="Train",
            use_container_width=True,
            type="primary",
        ):
            model = SpectrogramsModule(input_size, hidden_size, output_size)

            criterion = nn.MSELoss()

            optimizer = optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
            )

            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

            train_dataset = datasets.ImageFolder(
                root=dataset_root_dir,
                transform=transform,
            )

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

            placeholder_progress = st.empty()

            step_count = 0

            for epoch in range(num_epochs):
                for batch_idx, (data, _) in enumerate(train_loader):
                    data_flatten = data.view(-1)

                    outputs = model(data_flatten)
                    loss = criterion(outputs, data_flatten)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step_count += 1

                    placeholder_progress.progress(
                        value=step_count/ float(len(train_loader) * num_epochs),
                        text=f"Epoch {epoch + 1}/{num_epochs}, Step {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}",
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
                obj=model.state_dict(),
                f=model_output_path,
            )

            st.text("Done.")

            st.code(
                body=model_output_path,
                language="bash",
            )

