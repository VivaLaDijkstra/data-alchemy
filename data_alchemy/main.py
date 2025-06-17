import argparse
import sys

try:
    from megfile import SmartPath as Path
except ImportError:
    from pathlib import Path

from data_alchemy.tasks import get_task
from data_alchemy.utils.logging import logger
from data_alchemy.utils.string import parse_kwarg_str
from data_alchemy.utils.time import get_timestamp


def add_common_args(parser):
    parser.add_argument("--source-file", type=Path, nargs="+", required=True, help="source files")
    parser.add_argument("--output-file", type=Path, nargs="+", required=True, help="output files")
    parser.add_argument("--seed", type=int, required=False, default=42, help="random seed")
    parser.add_argument("--verbose", action="store_true", help="print more information")
    parser.add_argument("--debug", action="store_true", help="print debug information")
    parser.add_argument(
        "--dry-run", action="store_true", help="just run a few samples without writing to file"
    )


def setup_dedup_parser(subparsers):
    dedup_parser = subparsers.add_parser("dedup", help="deduplicate the dataset")
    add_common_args(dedup_parser)
    dedup_parser.add_argument(
        "--reference-file", type=Path, nargs="+", required=True, help="reference files to be used"
    )
    dedup_parser.add_argument(
        "--dedup-function",
        type=str,
        required=True,
        help="deduplication function name",
    )
    dedup_parser.add_argument("--use-gpu", action="store_true", help="use GPU for embedding")
    dedup_parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        help="similarity threshold",
    )
    dedup_parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        required=False,
        default=Path("~/.cache/data-alchemy/embeddings/").expanduser(),
        help="store embeddings cache in the directory",
    )
    dedup_parser.add_argument(
        "-m",
        "--embedding-model-name-or-path",
        type=str,
        required=False,
        help="embedding model name or path",
    )
    dedup_parser.add_argument(
        "--max-length", type=int, required=False, help="max input text length"
    )
    dedup_parser.add_argument(
        "--batch-size", type=int, required=False, default=64, help="embedding batch size"
    )


# TODO: not finished
def setup_analysis_parser(subparsers):
    analyze_parser = subparsers.add_parser("analyze", help="analyze the dataset")
    # context length analysis
    analyze_subparsers = analyze_parser.add_subparsers(dest="subcommand")
    length_parser = analyze_subparsers.add_parser("length", help="analyze the context length")
    length_parser.add_argument(
        "--bins",
        type=str,
        required=False,
        default="[1024*(2**i) for i in range(8)]",
        help="bins for length analysis",
    )
    length_parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        help="tokenizer name or path. if not provided, use the default tokenizer",
    )

    # conversation rounds analysis
    rounds_parser = analyze_subparsers.add_parser("rounds", help="analyze the conversation rounds")
    rounds_parser.add_argument(
        "--bins",
        type=str,
    )


def setup_translation_parser(subparsers):
    translation_parser = subparsers.add_parser("translate", help="translate the dataset")
    add_common_args(translation_parser)
    translation_parser.add_argument("--src-lang", type=str, required=True, help="source language")
    translation_parser.add_argument("--tgt-lang", type=str, required=True, help="target language")
    translation_parser.add_argument("--model", type=str, required=True, help="translation model")
    translation_parser.add_argument(
        "--max-tokens", type=int, required=False, default=1024, help="max tokens for output"
    )
    translation_parser.add_argument(
        "--num-workers", type=int, required=False, default=1, help="number of workers"
    )
    translation_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=False,
        default="./checkpoint/",
        help="checkpoint directory",
    )
    translation_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        required=False,
        default=9999999999,
        help="checkpoint interval",
    )
    translation_parser.add_argument(
        "--resume-from-checkpoint", action="store_true", help="resume from checkpoint"
    )
    translation_parser.add_argument(
        "--max-rerun-count",
        type=int,
        required=False,
        default=1,
        help="max rerun count for failed samples",
    )


def setup_sample_parser(subparsers):
    sample_parser = subparsers.add_parser("sample", help="sample a subset of dataset")
    add_common_args(sample_parser)
    # Create a mutually exclusive group for --ratio and --num
    group = sample_parser.add_mutually_exclusive_group()
    group.add_argument("--ratio", type=float, required=False, help="sampling ratio")
    group.add_argument("--num", type=int, required=False, help="number of samples")


def setup_merge_parser(subparsers):
    merge_parser = subparsers.add_parser("merge", help="merge multiple datasets")
    add_common_args(merge_parser)


def setup_sort_parser(subparsers):
    sort_parser = subparsers.add_parser("sort", help="sort a dataset")
    add_common_args(sort_parser)
    sort_parser.add_argument("--reverse", action="store_true", help="sort in reverse order")


def setup_split_parser(subparsers):
    split_parser = subparsers.add_parser("split", help="split a subset of dataset")
    add_common_args(split_parser)
    split_parser.add_argument(
        "--ratio",
        type=lambda s: [float(r.strip()) for r in s.split(",")],
        required=True,
        help="split ratio",
    )
    split_parser.add_argument("--randomly", action="store_true", help="randomly split the dataset")


def setup_generate_parser(subparsers):
    gen_parser = subparsers.add_parser("generate", help="generate data")
    add_common_args(gen_parser)
    gen_parser.add_argument(
        "--max-tokens", type=int, required=False, default=1024, help="max tokens"
    )
    gen_parser.add_argument(
        "--sample-process-function", type=str, required=True, help="sample process function"
    )
    gen_parser.add_argument(
        "--cache-dir",
        type=Path,
        required=False,
        default=Path("~/.cache/data-alchemy/model_call").expanduser(),
        help="cache directory",
    )
    gen_parser.add_argument(
        "--checkpoint-dir", type=Path, required=False, help="checkpoint directory"
    )
    gen_parser.add_argument(
        "--checkpoint-interval", type=int, required=False, help="checkpoint interval"
    )
    gen_parser.add_argument(
        "--resume-from-checkpoint", action="store_true", help="resume from checkpoint"
    )
    gen_parser.add_argument(
        "--num-workers", type=int, required=False, default=1, help="number of workers"
    )
    gen_parser.add_argument(
        "--max-rerun-count",
        type=int,
        required=False,
        default=0,
        help="max rerun count for failed samples",
    )


def setup_filter_parser(subparsers):
    filter_parser = subparsers.add_parser("filter", help="filter data")
    add_common_args(filter_parser)
    filter_parser.add_argument(
        "--sample-judge-function", type=str, required=True, help="sample judge function"
    )
    filter_parser.add_argument(
        "--partial-kwargs",
        type=str,
        required=False,
        default=None,
        help="sample judge function partial kwargs. e.g: 'tokenizer=Qwen/Qwen-14B, threshold=1024' for function 'is_overlong'",
    )


def setup_watch_parser(subparsers):
    watch_parser = subparsers.add_parser("watch", help="watch samples")
    watch_parser.add_argument(
        "--source-file", type=Path, required=True, nargs="+", help="source file to be watched"
    )
    watch_parser.add_argument("--randomly", action="store_true", help="randomly watch")
    watch_parser.add_argument("--show-raw", action="store_true", help="show raw samples")
    watch_parser.add_argument("--seed", type=int, required=False, default=0, help="random seed")


def setup_chat_parser(subparsers):
    chat_parser = subparsers.add_parser("chat", help="chat with LLMs")
    chat_parser.add_argument("--model", type=str, required=True, help="model name")
    chat_parser.add_argument("--system-prompt", type=str, required=False, help="system prompt")
    chat_parser.add_argument(
        "--frequency-penalty", type=float, required=False, help="frequency penalty"
    )
    chat_parser.add_argument(
        "--max-tokens", type=int, required=False, default=2048, help="max tokens for chat"
    )
    chat_parser.add_argument("--seed", type=int, required=False, help="random seed")
    chat_parser.add_argument("--stream", action="store_true", help="stream the response")
    chat_parser.add_argument("--stop", type=str, required=False, help="stop tokens")
    chat_parser.add_argument(
        "--temperature", type=float, required=False, default=0.0, help="temperature"
    )
    chat_parser.add_argument("--top-p", type=float, required=False, help="top p")
    chat_parser.add_argument(
        "--save-path", type=Path, required=False, help="save path for chat history"
    )


def post_process(args):
    # ensure source file exists
    if hasattr(args, "source_file"):
        for path in args.source_file:
            if not path.is_file():
                raise FileNotFoundError(f"source-file '{path}' does not exist")

    # ensure reference file exists
    if hasattr(args, "reference_file"):
        for path in args.reference_file:
            if not path.is_file():
                raise FileNotFoundError(f"reference-file '{path}' does not exist")

    # ensure output file exists
    if hasattr(args, "output_file"):
        for path in args.output_file:
            if path.is_dir():
                raise IsADirectoryError(f"Output file cannot be directory, got '{path}'")

        assert len({path.parent for path in args.output_file}) == 1, (
            "output files must in the same directory"
        )

        output_dir = args.output_file[0].parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

    if args.command in ("watch", "chat"):
        logger.debug(args)
        args.debug = False
        args.verbose = False
        return args

    if args.command in ("analysis"):
        # TODO: not finished
        args.check_list = [item.strip() for item in args.check_list.split(",")]

    if args.command in ("filter"):
        if args.partial_kwargs:
            args.partial_kwargs = parse_kwarg_str(args.partial_kwargs)

    logger.debug(args)
    return args


def get_args():
    parser = argparse.ArgumentParser(
        prog="data_alchemy",
        description="Data Alchemy, manufacture, clean, view, and debug your data!",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Register subparsers
    setup_dedup_parser(subparsers)
    setup_analysis_parser(subparsers)
    setup_translation_parser(subparsers)
    setup_sample_parser(subparsers)
    setup_merge_parser(subparsers)
    setup_sort_parser(subparsers)
    setup_split_parser(subparsers)
    setup_generate_parser(subparsers)
    setup_filter_parser(subparsers)
    setup_watch_parser(subparsers)
    setup_chat_parser(subparsers)

    args = parser.parse_args()
    args = post_process(args)

    return args


def main() -> int:
    args = get_args()
    kwargs = vars(args)
    command = kwargs.pop("command")
    debug = kwargs.pop("debug", False)
    verbose = kwargs.pop("verbose", False)

    if command not in ("watch", "chat"):
        level = "DEBUG" if debug else "INFO" if verbose else "WARNING"
        logger.remove()
        logger.add(sys.stdout, level=level)
        logger.add(
            f"logs/{command}_{get_timestamp()}.log",
            rotation="1 week",
            retention="120 days",
            level=level,
        )

    task_cls = get_task(command)
    task = task_cls(**kwargs)
    task.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
