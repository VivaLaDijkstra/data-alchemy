import re
from typing import Any, Iterable

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

# from tqdm.rich import tqdm
from tqdm import tqdm

from data_alchemy.datasets_ import BaseDataset
from data_alchemy.datasets_.base import Sample
from data_alchemy.datasets_.generic import GenericMessage, GenericSample
from data_alchemy.datasets_.utils import concat_datasets
from data_alchemy.tasks.base import BaseTask
from data_alchemy.utils.io import Path


class Watcher(BaseTask):
    """Watch for samples with nice UI"""

    def __init__(
        self,
        source_file: list[Path],
        randomly: bool = False,
        show_raw: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(
            task_name="watch",
            source_file=source_file,
            output_file=None,
            seed=seed,
            dry_run=False,
        )
        self.dataset: BaseDataset[Sample] = concat_datasets(self._load_datasets(source_file))
        self.schema = self.dataset.schema
        self.show_raw = show_raw

        self.console = Console(theme=Theme({"search_term": "bold magenta"}))
        if randomly:
            self.dataset = self.dataset.shuffle()

    def show_sample(self, id_: int, sample: Sample) -> None:
        message_panels = []
        sample = sample.to_generic()
        for i, msg in enumerate(sample.messages):
            md_content = Markdown(f"{msg.role}:\n{msg.content}")
            msg_panel = Panel(md_content, title=f"Message: {i}", border_style="yellow")
            message_panels.append(msg_panel)

        sample_panel = Panel(
            Columns(message_panels), title=f"Sample: {id_}", border_style="bold green"
        )

        if self.show_raw:
            self.console.print(sample)
        self.console.print(sample_panel)

    def search_sample(
        self, pattern: str, iterator: Iterable[Sample], pbar: tqdm, id_: int, sample: Sample
    ) -> Iterable[Sample]:
        """Search for samples with given pattern"""

        pbar.desc = f"Search for {pattern}"
        sample = sample.to_generic()

        res = None
        # for id_, sample in enumerate(iterator):
        for _ in range(len(self.dataset)):
            try:
                demo_messages: list[GenericMessage] = []
                for msg in sample.to_generic().messages:
                    res = re.search(pattern, msg.content, re.DOTALL)
                    if res:
                        demo_value = re.sub(
                            pattern,
                            f"==**__{res.group(0)}__**==",
                            msg.content,
                        )
                    else:
                        demo_value = msg.content

                    demo_messages.append(GenericMessage(role=msg.role, content=demo_value))

                if res:
                    demo_sample = GenericSample(messages=demo_messages)
                    self.show_sample(id_, demo_sample)
                    cmd = self.console.input(
                        f"Found '[bold green]{pattern}[/]' at sample {id_}."
                        f" Press 'n + Enter' to show next match or 'q + Enter' to quit search:\n"
                    )
                    if cmd.startswith("/"):
                        pattern = cmd[1:]
                    elif cmd == "n":
                        pass
                    elif cmd == "q":
                        pbar.desc = "scroll bar"
                        pbar.update(1)
                        return iterator
                    else:
                        pass

                id_, sample = next(iterator)
                pbar.update(1)
            except StopIteration:
                self.console.print(
                    "Searching hits the end of the dataset. Starting from beginning..."
                )
                pbar.reset()
                iterator = iter(enumerate(self.dataset))
                id_, sample = next(iterator)

        if not res:
            self.console.print(
                f"No match found for pattern '[bold magenta]{pattern}[/]'. Quit search."
            )
            pbar.desc = "scroll bar"

        return iterator

    def run(self) -> None:
        with tqdm(
            total=len(self.dataset),
            mininterval=0.01,
            maxinterval=0.02,
            desc="scroll bar",
            bar_format="{l_bar}{bar} [{n_fmt}/{total_fmt}]",
        ) as pbar:
            iterator = iter(enumerate(self.dataset))

            try:
                # get the first sample
                id_, sample = next(iterator)
                self.show_sample(id_, sample)
            except StopIteration:
                self.console.print("Dataset is empty")
                return
            except KeyboardInterrupt:
                self.console.print("Exiting...")
                return

            while True:
                try:
                    cmd = self.console.input(
                        "Press Enter to continue, or type /<regex> to search (ctrl+c to exit):\n"  # FIX: buffer flushed by pbar
                    )
                    if cmd.startswith("/"):
                        # search by given re pattern (vim style)
                        # search_iterator = circular(self.dataset, start=id_)
                        search_iterator = iterator
                        iterator = self.search_sample(cmd[1:], search_iterator, pbar, id_, sample)
                    else:
                        id_, sample = next(iterator)
                        self.show_sample(id_, sample)
                        pbar.update(1)
                except StopIteration:
                    cmd = self.console.input("Press Enter to restart from the beginning\n")
                    if cmd == "":
                        iterator = iter(enumerate(self.dataset))
                        pbar.reset()
                    else:
                        break
                except KeyboardInterrupt:
                    self.console.print("Exiting...")
                    break
