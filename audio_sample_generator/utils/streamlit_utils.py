from audio_sample_generator.data.sample_data import SampleData

import streamlit as st

from typing import List

SESSION_STATE_KEY_SAMPLE_DATA_LIST = "sample_data_list"

if st.session_state.get(SESSION_STATE_KEY_SAMPLE_DATA_LIST) is None:
    st.session_state[SESSION_STATE_KEY_SAMPLE_DATA_LIST] = []

sample_data_list: List[SampleData] = st.session_state[SESSION_STATE_KEY_SAMPLE_DATA_LIST]
