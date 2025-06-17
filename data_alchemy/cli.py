import os

import click
import rich
import rich.traceback
from loguru import logger

from data_alchemy.tasks import (
    Analyzer,
    DataGenerator,
    Deduplicator,
    Merger,
    Sampler,
    Sorter,
    Splitter,
    Translator,
)
from data_alchemy.utils.logging import logger
from data_alchemy.utils.time import get_timestamp

handler = rich.traceback.install()


def common_options(func):
    """Common options for all commands."""
    func = click.option("--source-file", required=True, help="Source files")(func)
    func = click.option("--output-file", required=True, help="Output file name")(func)
    func = click.option("--seed", default=42, help="Random seed")(func)
    func = click.option("--verbose", is_flag=True, help="Print more information")(func)
    func = click.option("--debug", is_flag=True, help="Print more information")(func)
    func = click.option("--dry-run", is_flag=True, help="Run a few samples, do not write to file")(
        func
    )
    return func


@click.group()
def cli():
    """Data Alchemy, manufacture, clean, view and debug your data!"""


def run_task(task_class, kwargs: dict[str, str | int | bool | float]):
    """Helper function to run tasks."""
    source_files = [path.strip() for path in kwargs.pop("source_file").split(",") if path.strip()]
    assert source_files, "source-file must be provided"
    for path in source_files:
        assert os.path.isfile(path), f"Source file {path} does not exist"

    output_files = [path.strip() for path in kwargs.pop("output_file").split(",") if path.strip()]
    assert output_files, "output-file must be provided"
    for path in output_files:
        assert not os.path.isdir(path), f"Output file cannot be directory, got {path}"

    kwargs["source_file"] = source_files
    kwargs["output_file"] = output_files

    command = kwargs.pop("command")
    debug = kwargs.pop("debug")
    verbose = kwargs.pop("verbose")

    logger.config(
        f"logs/{command}_{get_timestamp()}.log",
        rotation="1 week",
        retention="120 days",
        level=("DEBUG" if debug else "INFO" if verbose else "WARNING"),
    )

    logger.debug(kwargs)

    task = task_class(**kwargs)
    task.run()


@cli.command()
@common_options
@click.option("--reference-file", required=True, help="Reference files to be used")
@click.option(
    "--mode", default="similarity", help="Deduplication mode: 'exact', 'fuzzy', or 'similarity'"
)
@click.option("--embedding-model-name-or-path", help="Embedding model name or path")
@click.option("--max-length", default=512, help="Max length of the input text")
@click.option("--similarity-threshold", default=0.9, help="Similarity threshold for deduplication")
@click.option("--emb-batch-size", default=64, help="Batch size for embedding model")
@click.option("--use-gpu", is_flag=True, help="Use GPU for embedding model")
@click.option("--num-workers", default=1, help="Number of workers")
@click.option(
    "--store-embeddings-dir", default="./embeddings/", help="Path to save/load embeddings"
)
@click.option("--use-stored-embeddings", is_flag=True, help="Use stored embeddings")
def dedup(**kwargs):
    """Deduplicate the dataset."""
    run_task(Deduplicator, kwargs)


@cli.command()
@common_options
@click.option("--check-items", default="all", help="Data checking item list")
def analyze(**kwargs):
    """Analyze the dataset."""
    run_task(Analyzer, kwargs)


@cli.command()
@common_options
@click.option("--src-lang", required=True, help="Language to translate from")
@click.option("--tgt-lang", required=True, help="Language to translate to")
@click.option("--model", required=True, help="Model for translation")
@click.option("--max-tokens", default=1024, help="Maximum tokens for LLM output")
@click.option("--num-workers", default=0, help="Number of workers for LLM API call")
@click.option("--checkpoint-dir", default="./checkpoint/", help="Checkpoint directory")
@click.option(
    "--checkpoint-interval", default=9999999999, help="Save translated files every N samples"
)
@click.option("--resume-from-checkpoint", is_flag=True, help="Resume from checkpoint")
@click.option(
    "--max-rerun-count", default=-1, help="Max rerun count for failed samples (-1 = rerun forever)"
)
def translate(**kwargs):
    """Translate the dataset."""
    run_task(Translator, kwargs)


@cli.command()
@common_options
@click.option("--ratio", required=True, help="Sampling ratio")
def sample(**kwargs):
    """Sample a subset of the dataset."""
    run_task(Sampler, kwargs)


@cli.command()
@common_options
def merge(**kwargs):
    """Merge multiple datasets."""
    run_task(Merger, kwargs)


@cli.command()
@common_options
@click.option("--reverse", is_flag=True, help="Sort samples in reverse order")
def sort(**kwargs):
    """Sort the dataset."""
    run_task(Sorter, kwargs)


@cli.command()
@common_options
@click.option("--ratio", required=True, help="Splitting ratio")
@click.option("--randomly", is_flag=True, help="Randomly split the dataset")
def split(**kwargs):
    """Split the dataset."""
    run_task(Splitter, kwargs)


@cli.command()
def where():
    """Show current working directory."""
    rich.print(f"Current working directory: [bold green]{os.getcwd()}[/bold green]")


@cli.command()
@common_options
@click.option("--model", required=True, help="Model for data generation")
@click.option("--max-tokens", default=1024, help="Maximum tokens")
@click.option(
    "--sample-process-function",
    default="generate_last_turn",
    help="sample process function to be used (must be implemented in tasks/generation/process_functions)",
)
@click.option("--num-workers", default=0, help="Number of workers")
@click.option("--checkpoint-dir", default="./checkpoint/", help="Checkpoint directory")
@click.option(
    "--checkpoint-interval", default=9999999999, help="Save generated files every N samples"
)
@click.option("--resume-from-checkpoint", is_flag=True, help="Resume from checkpoint")
@click.option("--max-rerun-count", default=10, help="Max rerun count for failed samples")
def generate(**kwargs):
    """Generate data."""
    run_task(DataGenerator, kwargs)


if __name__ == "__main__":
    SystemExit(cli())
