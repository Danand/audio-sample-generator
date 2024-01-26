from audio_sample_generator.utils.image_utils import convert_mel_spectrogram_to_image
from audio_sample_generator.utils.streamlit_utils import sample_data_list

import streamlit as st

from typing import cast

title = "Prepare Dataset"

st.set_page_config(
    page_icon=title,
)

st.title(title)

if (len(sample_data_list) == 0):
    st.markdown("No samples here yet. But you can add at **Extract Spectrograms**.")
else:
    with st.container(border=True):
        st.subheader("Loaded Audio")

        for sample_data in sample_data_list:
            with st.container(border=True):
                is_enabled = st.checkbox(
                    label="Enabled",
                    value=True,
                    key=f"is_enabled_{sample_data.id}"
                )

                sample_data.is_enabled_for_training = is_enabled

                st.text(sample_data.input_audio_file_name)

                st.audio(
                    data=sample_data.waveform.numpy(),
                    sample_rate=sample_data.sample_rate,
                )

                mel_spectrogram_image = convert_mel_spectrogram_to_image(sample_data.mel_spectrogram)

                st.image(
                    image=mel_spectrogram_image,
                    output_format="PNG",
                )

                subject = st.text_input(
                    label="Subject",
                    placeholder="drum kit",
                    value=None,
                    key=f"subject_{sample_data.id}"
                )

                sample_data.subject = subject

                caption = st.text_area(
                    label="Caption",
                    placeholder="kick",
                    value=None,
                    key=f"caption_{sample_data.id}"
                )

                sample_data.caption = caption

                weight = cast(
                    float,
                    st.number_input(
                        label="Weight",
                        min_value=0.01,
                        value=1.0,
                        key=f"weight_{sample_data.id}"
                    )
                )

                sample_data.weight = weight

