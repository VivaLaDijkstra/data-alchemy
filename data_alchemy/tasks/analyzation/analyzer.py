from pathlib import Path


from data_alchemy.utils.io import rich_print


class Analyzer:
    def __init__(self, bins: list[int | float]) -> None:
        self.bins = bins

    def run(self) -> None:
        rich_print("Analyzing...", style="bold green")

