from sd_scripts.ui import (
    TIterableItem,
    UIWrapper,
)

from streamlit.delta_generator import DeltaGenerator

from typing import Iterable

class UIWrapperStreamlit(UIWrapper):
    def __init__(
        self,
        placeholder_progress: DeltaGenerator,
        iterable: Iterable[TIterableItem],
        desc: str,
    ):
        self.placeholder_progress = placeholder_progress
        self.total_steps = len(list(iterable))
        self.desc = desc

        self.postfix = ""
        self.current_step = 0

    def update(self, i: int = 1) -> None:
        self.current_step += i

        value = self.current_step / self.total_steps
        text = f"{self.desc}: [{self.current_step}/{self.total_steps}] {self.postfix}"

        self.placeholder_progress.progress(
            value=value,
            text=text,
        )

    def set_postfix(
        self,
        **kwargs,
    ) -> None:
        self.postfix = str(dict(**kwargs))

