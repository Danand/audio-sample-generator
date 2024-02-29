from audio_sample_generator.data.sample_data import SampleData
from audio_sample_generator.utils.image_utils import convert_mel_spectrogram_to_image
from audio_sample_generator.utils.streamlit_utils import sample_data_list
from audio_sample_generator.constants import DATASET_ROOT_DIR

import streamlit as st

from typing import Dict, List, cast
from os import makedirs
from shutil import rmtree
from math import floor, gcd

class DatasetFolderSaver:
    KEY_KOHYA_SS = "kohya_ss"

    OPTIONS = [
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

    def assign_weights(self, sample_data_list: List[SampleData]) -> None:
        raise NotImplementedError(f"Function `assign_weights` is not defined at `{self.__class__.__name__}`")

    def save(self, sample_data: SampleData, index: int) -> None:
        raise NotImplementedError(f"Function `save` is not defined at `{self.__class__.__name__}`")

    def collect_subjects(self, sample_data_list: List[SampleData]) -> None:
        for sample_data in sample_data_list:
            subject = sample_data.subject

            if subject not in self.subjects_collected and subject is not None:
                self.subjects_collected.append(subject)

    def prepare_parent_folder(self) -> None:
        rmtree(
            path=DATASET_ROOT_DIR,
            ignore_errors=True,
        )

class Subset:
    subject: str
    weight: float
    image_count: int
    repeats: int

def get_list_gcd(numbers: List[int]) -> int:
    result = numbers[0]

    for num in numbers[1:]:
        result = gcd(result, num)

    return result

def assign_repeats(subsets: List[Subset]) -> None:
    image_count_max = max(subsets, key=lambda subset: subset.image_count).image_count

    for subset in subsets:
        subset.repeats = floor((image_count_max * subset.weight) / subset.image_count)

    repeats_gcd = get_list_gcd([subset_for_subject.repeats for subset_for_subject in subsets])

    for subset in subsets:
        subset.repeats = floor(subset.repeats / float(repeats_gcd))

class DatasetFolderSaverKohyaSS(DatasetFolderSaver):
    subsets: List[Subset] = []
    subset_to_sample_data_ids: Dict[str, Subset] = {}

    @property
    def is_need_caption(self) -> bool:
        return True

    @property
    def is_need_weight(self) -> bool:
        return True

    def assign_weights(self, sample_data_list: List[SampleData]) -> None:
        for sample_data in sample_data_list:
            subset = next((
                    subset for subset in self.subsets
                    if subset.weight == sample_data.weight and
                       subset.subject == sample_data.subject
                ),
                None,
            )

            if subset is not None:
                subset.image_count +=1
            else:
                subset = Subset()

                subset.subject = cast(str, sample_data.subject)
                subset.weight = sample_data.weight
                subset.image_count = 1

                self.subsets.append(subset)

            self.subset_to_sample_data_ids[sample_data.id] = subset

        subjects_uniq = set([subset.subject for subset in self.subsets])

        for subject in subjects_uniq:
            subsets_for_subject = [subset for subset in self.subsets if subset.subject == subject]
            assign_repeats(subsets_for_subject)

    def save(self, sample_data: SampleData, index: int) -> None:
            subset = self.subset_to_sample_data_ids[sample_data.id]

            subset_dir = f"{DATASET_ROOT_DIR}/images/{subset.repeats}_{subset.subject}"

            makedirs(
                name=subset_dir,
                exist_ok=True,
            )

            image_name="image"

            mel_spectrogram_image_path = f"{subset_dir}/{image_name}-{index}.png"

            mel_spectrogram_image = convert_mel_spectrogram_to_image(sample_data.mel_spectrogram)

            mel_spectrogram_image.save(mel_spectrogram_image_path)

            st.text(f"Saved image for training: '{mel_spectrogram_image_path}'")

            caption_path = f"{subset_dir}/{image_name}-{index}.txt"

            if sample_data.caption is not None:
                with open(caption_path, "w") as file_caption:
                    file_caption.write(sample_data.caption)

                st.text(f"Saved caption for training: '{caption_path}'")

class DatasetFolderSaverFactory:
    @classmethod
    def create(cls, key: str) -> DatasetFolderSaver:
        return {
            DatasetFolderSaver.KEY_KOHYA_SS: lambda: DatasetFolderSaverKohyaSS(),
        }[key]()

title = "Prepare Dataset"

st.set_page_config(
    page_title=title,
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

                subject = st.text_input(
                    label="Subject",
                    value="DrumKit",
                    key=f"subject_{sample_data.id}"
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
        dataset_folder_saver.prepare_parent_folder()
        dataset_folder_saver.assign_weights(sample_data_list)

        with st.container(border=True):
            st.subheader("Saved Images Logs")

            for index, sample_data in enumerate(sample_data_list):
                dataset_folder_saver.save(sample_data, index)
