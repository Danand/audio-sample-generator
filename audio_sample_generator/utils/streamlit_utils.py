from audio_sample_generator.data.sample_data import SampleData
from audio_sample_generator.data.common_data import CommonData
from audio_sample_generator.utils.torch_utils import get_window_fn_dict

import streamlit as st

from typing import Callable, List, cast

SESSION_STATE_KEY_SAMPLE_DATA_LIST = "sample_data_list"

if st.session_state.get(SESSION_STATE_KEY_SAMPLE_DATA_LIST) is None:
    st.session_state[SESSION_STATE_KEY_SAMPLE_DATA_LIST] = []

sample_data_list: List[SampleData] = st.session_state[SESSION_STATE_KEY_SAMPLE_DATA_LIST]

SESSION_STATE_KEY_COMMON_DATA = "common_data"

if st.session_state.get(SESSION_STATE_KEY_COMMON_DATA) is None:
    st.session_state[SESSION_STATE_KEY_COMMON_DATA] = CommonData()

common_data: CommonData = st.session_state[SESSION_STATE_KEY_COMMON_DATA]

class SpectrogramToAudioSettings:
    sample_rate: int
    f_min: float
    f_max: float | None
    norm: str | None
    n_fft: int
    n_iter: int
    win_length: int | None
    hop_length: int
    n_mels: int
    window_fn: Callable
    mel_scale: str
    driver: str
    power: float
    momentum: float
    length: int | None
    rand_init: bool

def get_spectrogram_to_audio_settings() -> SpectrogramToAudioSettings:
    st.subheader("Mel Spectrogram Settings")

    settings = SpectrogramToAudioSettings()

    settings.sample_rate = cast(
        int,
        st.selectbox(
            label="Sample rate",
            options=[
                8000,
                11025,
                16000,
                22050,
                44100,
                48000,
            ],
            index=3,
        )
    )

    settings.f_min = cast(
        float,
        st.number_input(
            label="Minimum Frequency",
            min_value=0.0,
            value=0.0,
        )
    )

    custom_value_enabled_f_max = st.checkbox(
        label="Custom Maximum Frequency",
        value=False,
    )

    settings.f_max = cast(
        float,
        st.number_input(
            label="Maximum Frequency",
            min_value=0.0,
            value=float(settings.sample_rate // 2),
            step=1.0,
            disabled=not custom_value_enabled_f_max,
        )
    )

    if not custom_value_enabled_f_max:
        settings.f_max = None

    settings.norm = st.selectbox(
        label="Area Normalization",
        options=[
            None,
            "slaney",
        ],
        index=0,
    )

    settings.n_fft = cast(
        int,
        st.number_input(
            label="FFT Length",
            min_value=0,
            value=400,
        )
    )

    settings.n_iter = cast(
        int,
        st.number_input(
            label="Phase Recovery Iterations",
            min_value=1,
            value=32,
        )
    )

    custom_value_enabled_win_length = st.checkbox(
        label="Custom Window Length",
        value=False,
    )

    settings.win_length = cast(
        int,
        st.number_input(
            label="Window Length",
            min_value=0,
            value=400,
            disabled=not custom_value_enabled_win_length,
        )
    )

    if not custom_value_enabled_win_length:
        settings.win_length = None

    settings.hop_length = cast(
        int,
        st.number_input(
            label="Hop Length",
            min_value=0,
            value=400,
        )
    )

    settings.n_mels = cast(
        int,
        st.number_input(
            label="Mel filter banks number",
            min_value=0,
            value=128,
        )
    )

    window_fn_dict = get_window_fn_dict()

    window_fn_key = cast(
        str,
        st.selectbox(
            label="Window Function",
            options=window_fn_dict.keys(),
            index=0,
        )
    )

    settings.window_fn = window_fn_dict[window_fn_key]

    settings.mel_scale = cast(
        str,
        st.selectbox(
            label="Mel Scale",
            options=[
                "htk",
                "slaney",
            ],
            index=0,
        )
    )

    settings.driver = cast(
        str,
        st.selectbox(
            label="Driver",
            options=[
                "gels",
                "gelsy",
                "gelsd",
                "gelss",
            ],
            index=1,
        )
    )

    settings.power = cast(
        float,
        st.number_input(
            label="Power",
            min_value=0.0,
            value=1.0,
        )
    )

    settings.momentum = cast(
        float,
        st.number_input(
            label="Momentum",
            min_value=0.0,
            value=0.99,
            help="The momentum parameter for fast Griffin-Lim. Setting this to 0 recovers the original Griffin-Lim method. Values near 1 can lead to faster convergence, but above 1 may not converge.",
        )
    )

    custom_value_enabled_length = st.checkbox(
        label="Custom Length",
        value=False,
    )

    settings.length = cast(
        int,
        st.number_input(
            label="Length",
            min_value=0,
            value=int(settings.sample_rate / settings.hop_length),
            disabled=not custom_value_enabled_length,
        )
    )

    if not custom_value_enabled_length:
        settings.length = None

    settings.rand_init = st.checkbox(
        label="Random Phase",
        value=True,
    )

    return settings
