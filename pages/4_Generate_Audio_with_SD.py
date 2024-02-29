from audio_sample_generator.utils.streamlit_utils import get_spectrogram_to_audio_settings
from audio_sample_generator.utils.torch_utils import get_available_devices

import streamlit as st

from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

import torch
import torchaudio
import torchvision

from PIL import Image

from audio_sample_generator import constants

import time

from posixpath import basename, dirname
from typing import cast, List
from math import ceil

title = "Generate Audio with Stable Diffusion"

st.set_page_config(
    page_icon=title,
)

st.title(title)

with st.container(border=True):
    settings = get_spectrogram_to_audio_settings()

with st.container(border=True):
    st.subheader("Stable Diffusion Settings")

    base_model = st.text_input(
        label="Base Model",
        value=constants.DEFAULT_BASE_MODEL,
    )

    lora_path = st.text_input(
        label="LoRA Path",
        value=f"{constants.DATASET_ROOT_DIR}/output/{constants.OUTPUT_MODEL_NAME}.safetensors",
    )

    prompt = st.text_area(
        label="Prompt",
        placeholder="808, kick",
    )

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

    num_inference_steps = cast(
        int,
        st.number_input(
            label="Steps",
            min_value=1,
            value=20,
        ),
    )

    guidance_scale = cast(
        float,
        st.number_input(
            label="Guidance Scale",
            min_value=0.0,
            max_value=100.0,
            value=7.0,
        )
    )

    lora_scale = cast(
        float,
        st.number_input(
            label="LoRA Scale",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
        )
    )

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

    generate_amount_total = cast(
        int,
        st.number_input(
            label="Amount",
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

if st.button(
    label="Generate",
    use_container_width=True,
    type="primary",
):
    with st.container(border=True):
        st.subheader("Generated Audio")

        with st.spinner("Loading pipeline..."):
            pipeline = AutoPipelineForText2Image.from_pretrained(
                pretrained_model_or_path=base_model,
                torch_dtype=torch.float32,
                safety_checker = None,
                requires_safety_checker = False,
            ).to(device_name)

        pipeline = cast(StableDiffusionPipeline, pipeline)

        lora_dir = dirname(lora_path)
        lora_name = basename(lora_path)

        with st.spinner("Loading LoRA..."):
            pipeline.load_lora_weights(
                pretrained_model_name_or_path_or_dict=lora_dir,
                weight_name=lora_name,
            )

            pipeline.fuse_lora(
                lora_scale=lora_scale,
            )

        generate_amount_total_max = max(
            ceil(generate_amount_total / batch_size) * batch_size,
            batch_size,
        )

        generate_amount = generate_amount_total_max // batch_size

        images: List[Image.Image] = []

        seed_offset = 0

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

        time_start = time.time()

        with st.spinner("Generating spectrograms..."):
            placeholder_progress_images = st.empty()
            placeholder_progress_steps = st.empty()

            for generating_image_index in range(generate_amount):
                def on_progress_steps(step: int, timestep: int, tensor: torch.FloatTensor | None):
                    placeholder_progress_steps.progress(
                        value=step / float(num_inference_steps),
                        text=f"Steps completed: {step}/{num_inference_steps}",
                    )

                placeholder_progress_images.progress(
                    value=len(images) / float(generate_amount_total_max),
                    text=f"Images completed: {len(images)}/{generate_amount_total_max}",
                )

                generator = [
                    torch.Generator(device=device_name).manual_seed(seed + seed_offset + generating_image_index + batch_image_index)
                    for batch_image_index in range(batch_size)
                ]

                seed_offset += (batch_size - 1)

                output = cast(
                    StableDiffusionPipelineOutput,
                    pipeline(
                        prompt=prompt,
                        generator=generator,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=batch_size,
                        callback=on_progress_steps,
                        callback_steps=1,
                    )
                )

                for image in output.images:
                    images.append(image)

                placeholder_progress_images.progress(
                    value=len(images) / float(generate_amount_total_max),
                    text=f"Images completed: {len(images)}/{generate_amount_total_max}",
                )

        time_end = time.time()
        time_elapsed = time_end - time_start

        st.text("Generation of spectrograms completed.")
        st.text(f"Time elapsed: {time_elapsed:.2f} s.")

        for image in images:
            with st.container(border=True):
                with st.spinner("Extraction audio from spectrogram..."):
                    mel_spectrogram = transform_image_tensor(image).squeeze(0) * 255
                    lin_spectrogram = transform_inverse_mel_scale(mel_spectrogram)
                    waveform = transform_griffin_lim(lin_spectrogram)

                st.audio(
                    data=waveform.numpy(),
                    sample_rate=settings.sample_rate,
                )

                st.image(
                    image=image,
                    use_column_width=True,
                )

