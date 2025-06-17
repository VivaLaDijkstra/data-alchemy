import json
import os
from io import IOBase
from typing import IO, TYPE_CHECKING, Any, Iterable, Literal, Optional

try:
    from megfile import SmartPath as Path
    from megfile import smart_open as file_open
except ImportError:
    from pathlib import Path

    file_open = open

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

import ijson
import rich
from rich.columns import Columns
from rich.errors import ConsoleError, StyleError
from rich.markdown import Markdown
from rich.panel import Panel

from data_alchemy.datasets_.base import RawSample, RootSample, Sample
from data_alchemy.utils.logging import logger

ONE_GIGA_BYTES: int = 2**32


def rich_print(
    *objects: Any,
    sep: str = " ",
    end: str = "\n",
    file: Optional[IO[str] | "SupportsWrite[str]"] = None,
    flush: Literal[False] = False,
) -> None:
    try:
        rich.print(*objects, sep=sep, end=end, file=file, flush=flush)
    except (ConsoleError, StyleError, AssertionError):
        print(*objects, sep=sep, end=end, file=file, flush=flush)


def extract_file_name(fpath: str) -> str:
    """Extract file name from file path."""
    return os.path.splitext(os.path.basename(fpath))[0]


def get_file_byte_length(file_handler: IOBase) -> int:
    file_handler.seek(0, 2)
    bytes_lengths = file_handler.tell()
    file_handler.seek(0)
    return bytes_lengths


def load_first_sample(fpath: str | Path) -> RawSample:
    """Load the 1st sample from json file."""

    with file_open(fpath, mode="r", encoding="utf-8") as f:
        try:
            return next(ijson.items(f, "item", use_float=True))
        except StopIteration as e:
            logger.error(f"`{fpath}` has 0 sample")
            raise e
        except ijson.IncompleteJSONError as e:
            logger.error(f"`{fpath}` contains incomplete JSON")
            raise e


def load_samples(fpath: str | Path) -> list[RawSample]:
    fpath = Path(Path(fpath).expanduser())

    if fpath.suffix == ".json":
        with file_open(fpath, mode="r", encoding="utf-8") as f:
            return list(ijson.items(f, "item", use_float=True))
    elif fpath.suffix == ".jsonl":
        with file_open(fpath, mode="r", encoding="utf-8") as f:
            return list(ijson.items(line, "item", use_float=True) for line in f)
    else:
        raise ValueError(f"Unsupported sample file type: {fpath.suffix}")


def dump_samples(samples: Iterable[Sample], fpath: str | Path, mode: str = "w+"):
    """Dump samples to json file."""
    with file_open(fpath, mode=mode, encoding="utf-8") as f:
        decapsulated_samples: list[RawSample] = [smp.to_raw() for smp in samples]
        json.dump(decapsulated_samples, f, ensure_ascii=False, indent=4)


def show_sample(sample: RootSample, title: str = "Sample") -> None:
    message_panels = []
    messages = sample.root

    for i, msg in enumerate(messages):
        if msg.role:
            role = msg.role
        else:
            role = ""
            for piece in msg.content:
                if isinstance(piece, str):
                    role += piece
                elif isinstance(piece, dict):
                    role += piece["role"]
                else:
                    raise TypeError(f"Unknown type of piece: {piece}")
        md_content = Markdown(f"{msg.role}:\n{role}")
        msg_panel = Panel(md_content, title=f"Message: {i}", border_style="yellow")
        message_panels.append(msg_panel)

    sample_panel = Panel(
        Columns(message_panels), title=title, border_style="bold green"
    )
    rich_print(sample_panel)
