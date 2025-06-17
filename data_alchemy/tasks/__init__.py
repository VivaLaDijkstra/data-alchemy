from .analyzation import Analyzer
from .chat import Chatter
from .deduplication import Deduplicator
from .filtration import Filter
from .generation.generator import DataGenerator
from .merge import Merger
from .sample import Sampler
from .sort import Sorter
from .split import Splitter
from .translate import Translator
from .watch import Watcher

__all__ = [
    "Analyzer",
    "Chatter",
    "DataGenerator",
    "Deduplicator",
    "Merger",
    "Sampler",
    "Sorter",
    "Splitter",
    "Translator",
    "Watcher",
]

task_map = {
    "analyze": Analyzer,
    "chat": Chatter,
    "dedup": Deduplicator,
    "filter": Filter,
    "generate": DataGenerator,
    "merge": Merger,
    "sample": Sampler,
    "sort": Sorter,
    "split": Splitter,
    "translate": Translator,
    "watch": Watcher,
}


def get_task(task_name: str):
    """Get the task class by name."""
    return task_map[task_name]
