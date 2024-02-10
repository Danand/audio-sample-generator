from audio_sample_generator.data.sample_data import SampleData
from audio_sample_generator.utils.image_utils import convert_mel_spectrogram_to_image
from audio_sample_generator.utils.streamlit_utils import sample_data_list
from audio_sample_generator.constants import DATASET_ROOT_DIR

import streamlit as st

from typing import List, cast
from os import makedirs
from shutil import rmtree

class DatasetFolderSaver:
    KEY_PLAIN_PYTORCH = "Plain PyTorch"
    KEY_KOHYA_SS = "kohya_ss"

    OPTIONS = [
        KEY_PLAIN_PYTORCH,
        KEY_KOHYA_SS,
    ]

    subjects_collected = []

    @property
    def is_need_caption(self) -> bool:
        raise NotImplementedError(f"Property `is_need_caption` is not defined at `{self.__class__.__name__}`")

    @property
    def is_need_weight(self) -> bool:
        raise NotImplementedError(f"Property `is_need_weight` is not defined at `{self.__class__.__name__}`")

    @property
    def subjects(self) -> List[str]:
        return self.subjects_collected

    def save(self, sample_data: SampleData) -> None:
        raise NotImplementedError(f"Function `save` is not defined at `{self.__class__.__name__}`")

    def collect_subjects(self, sample_data_list: List[SampleData]) -> None:
        for sample_data in sample_data_list:
            subject = sample_data.subject

            if subject not in self.subjects_collected and subject is not None:
                self.subjects_collected.append(subject)

class DatasetFolderSaverPlainPyTorch(DatasetFolderSaver):
    @property
    def is_need_caption(self)-> bool:
        return False

    @property
    def is_need_weight(self) -> bool:
        return False

    def save(self, sample_data: SampleData) -> None:
        rmtree(
            path=DATASET_ROOT_DIR,
            ignore_errors=True,
        )

        class_name = "spectrograms" if sample_data.subject is None else sample_data.subject

        class_dir = f"{DATASET_ROOT_DIR}/{class_name}"

        makedirs(
            name=class_dir,
            exist_ok=True,
        )

        mel_spectrogram_image_path = f"{class_dir}/{sample_data.id}.png"

        mel_spectrogram_image = convert_mel_spectrogram_to_image(sample_data.mel_spectrogram)

        mel_spectrogram_image.save(mel_spectrogram_image_path)

        st.text(f"Saved image for training: '{mel_spectrogram_image_path}'")

class DatasetFolderSaverKohyaSS(DatasetFolderSaver):
    @property
    def is_need_caption(self) -> bool:
        return True

    @property
    def is_need_weight(self) -> bool:
        return True

class DatasetFolderSaverFactory:
    @classmethod
    def create(cls, key: str) -> DatasetFolderSaver:
        return {
            DatasetFolderSaver.KEY_PLAIN_PYTORCH: lambda: DatasetFolderSaverPlainPyTorch(),
            DatasetFolderSaver.KEY_KOHYA_SS: lambda: DatasetFolderSaverKohyaSS(),
        }[key]()

title = "Prepare Dataset"

st.set_page_config(
    page_icon=title,
)

st.title(title)

if (len(sample_data_list) == 0):
    st.markdown("No samples here yet. But you can add at **Extract Spectrograms**.")
else:
    with st.container(border=True):
        st.subheader("Common Settings")

        dataset_folders_layout = cast(
            str,
            st.selectbox(
                label="Dataset Folders Layout",
                options=DatasetFolderSaver.OPTIONS,
                index=0,
            )
        )

    dataset_folder_saver = DatasetFolderSaverFactory.create(
        key=dataset_folders_layout,
    )

    with st.container(border=True):
        st.subheader("Loaded Audio")

        dataset_folder_saver.collect_subjects(sample_data_list)

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
                    image=mel_spectrogram_image.convert("RGB"),
                    output_format="PNG",
                    use_column_width="always",
                )

                selected_subject = st.selectbox(
                    label="Subject",
                    options=[ "Custom", *dataset_folder_saver.subjects ],
                    index=0,
                    key=f"subject_{sample_data.id}",
                )

                if selected_subject in dataset_folder_saver.subjects:
                    subject = selected_subject
                else:
                    subject = st.text_input(
                        label="Custom Subject",
                        value="DrumKit",
                        key=f"subject_custom_{sample_data.id}"
                    )

                sample_data.subject = subject

                if dataset_folder_saver.is_need_caption:
                    caption = st.text_area(
                        label="Caption",
                        placeholder="kick",
                        value=None,
                        key=f"caption_{sample_data.id}"
                    )

                    sample_data.caption = caption

                if dataset_folder_saver.is_need_weight:
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

    if st.button(
        label="Save",
        use_container_width=True,
        type="primary",
    ):
        with st.container(border=True):
            st.subheader("Saved Images Logs")

            for sample_data in sample_data_list:
                dataset_folder_saver.save(sample_data)
