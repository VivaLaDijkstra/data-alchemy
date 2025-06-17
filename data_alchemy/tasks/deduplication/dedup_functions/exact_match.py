import json
from typing import Any

from tqdm.rich import tqdm

from data_alchemy.datasets_.base import BaseDataset, Sample


def to_json_str(smp: Sample) -> str:
    raw: dict[str, Any] = smp.to_str()
    return json.dumps(raw, sort_keys=True)


def exact_match(
    source: BaseDataset[Sample], reference: BaseDataset[Sample], **kwargs
) -> BaseDataset[Sample]:
    reference_hashes = {
        to_json_str(smp) for smp in tqdm(reference, desc="Preparing reference hashes")
    }

    return BaseDataset(
        [
            smp
            for smp in tqdm(source, desc="Exact Match deduplicating")
            if to_json_str(smp) not in reference_hashes
        ],
        schema=source.schema,
    )
