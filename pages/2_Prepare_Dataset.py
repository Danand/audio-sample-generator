from audio_sample_generator.utils.streamlit_utils import sample_data_list

import streamlit as st

title = "Prepare Dataset"

st.set_page_config(
    page_icon=title,
)

st.title(title)

if (len(sample_data_list) == 0):
    st.markdown("No samples here yet. But you can add at **Extract Spectrograms**.")

for sample_data in sample_data_list:
    st.audio(
        data=sample_data.waveform.numpy(),
        sample_rate=sample_data.sample_rate,
    )

    # TODO: Fill captions.
