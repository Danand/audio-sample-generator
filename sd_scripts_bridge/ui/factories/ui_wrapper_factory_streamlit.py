from sd_scripts.ui import (
    TIterableItem,
    UIWrapper,
    UIWrapperFactory,
)

from ..wrappers.ui_wrapper_streamlit import UIWrapperStreamlit

from streamlit.delta_generator import DeltaGenerator

from typing import Iterable

class UIWrapperFactoryStreamlit(UIWrapperFactory):
    def __init__(self, placeholder_progress: DeltaGenerator):
        self.placeholder_progress = placeholder_progress

    def create(
        self,
        iterable: Iterable[TIterableItem],
        smoothing: float,
        disable: bool,
        desc: str,
    ) -> UIWrapper:
        return UIWrapperStreamlit(
            placeholder_progress=self.placeholder_progress,
            iterable=iterable,
            desc=desc,
        )

