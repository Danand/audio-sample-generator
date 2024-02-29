import sys

sys.path.append("sd_scripts")

from audio_sample_generator import constants
from audio_sample_generator.utils.streamlit_utils import common_data

from sd_scripts_bridge.ui.factories.ui_wrapper_factory_streamlit import UIWrapperFactoryStreamlit

import streamlit as st

from PIL import Image

from sd_scripts.train_network import NetworkTrainer, setup_parser
from sd_scripts.library.train_util import read_config_from_file

from argparse import Namespace
from typing import cast, Tuple
from glob import glob
from os.path import basename

def calc_steps(train_data_dir: str) -> int:
    steps_total = 0

    subset_dirs = glob(f"{train_data_dir}/*_*")

    for subset_dir in subset_dirs:
        subset_folder = basename(subset_dir)
        repeats = int(subset_folder.split("_")[0])
        image_paths = glob(f"{subset_dir}/*.png") + glob(f"{subset_dir}/*.jpg")

        steps_total += repeats * len(image_paths)

    return steps_total

def calc_size_max(train_data_dir: str) -> Tuple[int, int]:
    width_max = 0
    height_max = 0

    image_paths = glob(f"{train_data_dir}/**/**/*.png") + glob(f"{train_data_dir}/**/**/*.jpg")

    for image_path in image_paths:
        image = Image.open(image_path)

        width_max = max(width_max, image.width)
        height_max = max(height_max, image.height)

    return width_max, height_max

title = "Train LoRA"

st.set_page_config(
    page_icon=title,
)

st.title(title)

with st.container(border=True):
    st.subheader("Train LoRA Settings")

    base_model = st.text_input(
        label="Base Model",
        value=constants.DEFAULT_BASE_MODEL,
    )

    training_dir = st.text_input(
        label="Training Directory",
        value=constants.DATASET_ROOT_DIR,
    )

    output_name = st.text_input(
        label="Output Model Name",
        value=constants.OUTPUT_MODEL_NAME,
    )

    common_data.output_model_name = constants.OUTPUT_MODEL_NAME

    seed = cast(
        int,
        st.number_input(
            label="Seed",
            value=808,
            step=1,
            format="%i",
            min_value=constants.INT_MIN_VALUE,
            max_value=constants.INT_MAX_VALUE,
        ),
    )

    width_max, height_max = calc_size_max(training_dir)

    width = cast(
        int,
        st.number_input(
            label="Width",
            value=width_max,
            step=1,
            format="%i",
        ),
    )

    height = cast(
        int,
        st.number_input(
            label="Height",
            value=height_max,
            step=1,
            format="%i",
        ),
    )

    epochs = cast(
        int,
        st.number_input(
            label="Epochs",
            value=15,
            step=1,
            format="%i",
        ),
    )

    network_dim = cast(
        int,
        st.number_input(
            label="Network Dimension",
            value=128,
            step=1,
            format="%i",
        ),
    )

    network_alpha = cast(
        int,
        st.number_input(
            label="Network Alpha",
            value=128,
            step=1,
            format="%i",
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

    unet_lr = cast(
        float,
        st.number_input(
            label="UNet Learning Rate",
            min_value=0.0000001,
            value=0.0001,
            step=0.0000001,
            format="%.7f",
        ),
    )

    text_encoder_lr = cast(
        float,
        st.number_input(
            label="UNet Learning Rate",
            min_value=0.0000001,
            value=5e-5,
            step=0.0000001,
            format="%.7f",
        ),
    )

    noise_offset = cast(
        float,
        st.number_input(
            label="Noise Offset",
            min_value=0.0,
            value=0.1,
            step=0.01,
            format="%.2f",
        ),
    )

    if st.button(
        label="Train",
        use_container_width=True,
        type="primary",
    ):
        train_data_dir = f"{training_dir}/images"
        max_train_steps = calc_steps(train_data_dir) * epochs

        args_predefined = Namespace(
            pretrained_model_name_or_path=base_model,
            train_data_dir=train_data_dir,
            output_dir=f"{training_dir}/output",
            logging_dir=f"{training_dir}/logs",
            output_name=output_name,
            resolution=f"{width},{height}",
            seed=seed,
            lr_scheduler_num_cycles=epochs,
            network_dim=network_dim,
            network_alpha=network_alpha,
            learning_rate=learning_rate,
            unet_lr=unet_lr,
            text_encoder_lr=text_encoder_lr,
            max_train_steps=max_train_steps,
            save_model_as="safetensors",
            network_module="networks.lora",
            no_half_vae=True,
            lr_scheduler="cosine_with_restarts",
            train_batch_size=1,
            save_every_n_epochs=epochs,
            mixed_precision="no",
            save_precision="float",
            caption_extension=".txt",
            cache_latents=True,
            cache_latents_to_disk=True,
            optimizer_type="AdamW",
            max_data_loader_n_workers=0,
            gradient_checkpointing=True,
            bucket_no_upscale=True,
            noise_offset=noise_offset,
            network_train_unet_only=True,
            lowram=True,
        )

        parser = setup_parser()

        args_parsed = parser.parse_args(namespace=args_predefined)
        args_enriched = read_config_from_file(args_parsed, parser)

        placeholder_progress = st.empty()
        ui_wrapper_factory = UIWrapperFactoryStreamlit(placeholder_progress)

        with st.spinner("Training..."):
            trainer = NetworkTrainer()
            trainer.train(args_enriched, ui_wrapper_factory)

        st.text("Done.")

