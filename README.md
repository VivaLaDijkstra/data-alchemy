# Data Alchemy

Transmuting data to make life better\~

## Intro

Overwhelmed by the flood of generated data?
What if your script crashes halfway and needs to rerun?
Files too large to open without freezing?

If you've ever faced these soul-crushing dilemmas, then you need:

> **Data Alchemy**

Generate data, deduplicate, perform quality checks, split datasets into ablation sets...
Whatever your data-related needs are, Data Alchemy has you covered! No more staying late writing scattered scripts—your productivity just got supercharged.

## Features

* **Automatic fault tolerance**, resume from breakpoints, efficient API retries—reliable results without worrying about low-level issues
* **CPU/GPU parallelism**—blazing fast so you can finish work early and go home to play *Black Myth: Wukong*
* **Automatic sample schema matching**—compatible with both open-source and internal company formats
* **Streaming file I/O**—supports OSS, opens massive datasets instantly, no more staring at a frozen Vim
* **Comprehensive logging**—track every failure event
* **Highly extensible**—tailor it to your unique workflow with ease
* **One-click install, ready to use**—and if you get stuck, a charming developer is just a ping away

## Developing

* **Data Generation**
* **Data Cleaning**

  * \[✅] Semantic deduplication with BGE-m3
  * [ ] KNN-based deduplication
  * [ ] Jaccard similarity & MinHash deduplication
* **Data Visualization**

  * \[✅] Command-line sample highlighting
  * [ ] T-SNE visualization

You can easily extend functionality to fit your custom needs.

## Development Guide

Things you need to know for customized development:

### Project Structure

```bash
.
├── demo/  # Demo files
├── scripts/  # Ready-to-use scripts
├── data_alchemy/  # Source code
│   ├── datasets_/  # **Datasets**, the fundamental units manipulated by data-alchemy
│   │   ├── base.py  # Base class defining datasets, samples, messages, and core dataset functionality
│   │   ├── generic.py  # Central data format that can convert to/from all other types
│   │   ├── oai.py  # Data types for OpenAI APIs
│   │   └── *.py  # Other specific dataset types
│   ├── models/  # Models
│   │   ├── base.py  # Base model interface
│   │   ├── embedding/  # Embedding models
│   │   │   ├── base.py  # Embedding model interface
│   │   │   ├── bge.py  # BGE model implementation
│   │   │   └── *.py  # Other embedding models
│   │   ├── generative/  # Generative models
│   │   │   ├── base.py  # Generative model interface
│   │   │   ├── oai.py  # OpenAI API interface
│   │   │   ├── anthropic.py  # Anthropic API interface
│   │   │   └── *.py  # Other generative models
│   ├── tasks/  # Supported tasks
│   │   ├── base.py  # Base task class with support for checkpointing, retries, etc.
│   │   ├── analyzation/
│   │   ├── generation/  # Data generation (e.g., single/multi-turn dialogue)
│   │   ├── deduplication/  # Dataset deduplication
│   │   ├── merging.py  # Merge datasets
│   │   ├── sampling.py  # Dataset sampling
│   │   ├── sorting.py  # Sort datasets, supports custom keys
│   │   ├── splitting.py  # Split datasets
│   │   └── translation.py  # Dataset translation
│   ├── visualization/
│   └── utils/
└── tests/  # Unit tests
    └── data_alchemy/
```

### Custom Data Generation Logic

Override `process_sample` to implement custom generation logic (e.g., MCTS, BoN):

```python
# data_alchemy/tasks/generation/generation.py
class DataGenerator(GenerativeTask):
    ...
    def process_sample(self, id_: int, source: Sample) -> tuple[int, Sample]:
        """
        Args:
            id_ (int): Keeps sample order for multiprocessing
            source (RootSample): Source sample to process

        Returns:
            tuple[int, Sample]: ID and processed sample
        """
        ...
```

### Adding New Dataset Types

Open-source datasets often differ from internal formats. You'll need to create custom dataset classes, sample schemas, and utility interfaces:

```python
# data_alchemy/datasets_/my_new_dataset.py
from .base import BaseDataset, BaseSample
from .generic import GenericMessage, GenericSample

class MyNewSample(BaseSample):
    problem: str
    level: str
    solution: str

    def to_str(self) -> str:
        return f"{self.problem}\n{self.solution}"

    def to_generic(self) -> GenericSample:
        return GenericSample(
            messages=[
                GenericMessage(role="user", content=self.problem),
                GenericMessage(role="assistant", content=self.solution),
            ]
        )


class MyNewDataset(BaseDataset):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path, schema=MyNewSample)
```

### Contribution Guidelines

1. Pull from the `dev` branch and create a new branch for your work.
2. After development, submit a merge request back to `dev`, and assign `wuhanghao` as the reviewer.
3. Follow code style using `black`, `ruff`, and `isort`.
4. **Unit tests**: Required in `tests/data_alchemy/` (preferably using `pytest`). Aim for at least 80% coverage.
