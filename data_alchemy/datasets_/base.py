import random
from abc import ABC, abstractmethod

try:
    from megfile import SmartPath as Path
except ImportError:
    from pathlib import Path

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    KeysView,
    Sequence,
    TypeAlias,
    TypeVar,
)

from ordered_set import OrderedSet
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError, conlist
from tqdm.rich import tqdm

from data_alchemy.utils.logging import logger

if TYPE_CHECKING:
    from .generic import GenericMessage, GenericSample

RawSample: TypeAlias = dict[str, Any] | list[dict[str, Any]]


class BaseMessage(ABC, BaseModel):
    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    def to_raw(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)

    @abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def to_generic(self) -> "GenericMessage":
        raise NotImplementedError("Subclass must implement this method")


class BaseSample(ABC, BaseModel):
    model_config = ConfigDict(extra="ignore")

    def to_raw(self) -> RawSample:
        return self.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)

    @abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def to_generic(self) -> "GenericSample":
        raise NotImplementedError("Subclass must implement this method")


class RootSample(ABC, RootModel):
    if TYPE_CHECKING:
        root: list[BaseMessage]
    else:
        root: conlist(BaseMessage, min_length=1)

    def __getitem__(self, idx: int) -> BaseMessage:
        return self.root[idx]

    def to_raw(self) -> list[dict[str, Any]]:
        return self.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)

    @abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def to_generic(self) -> "GenericSample":
        raise NotImplementedError("Subclass must implement this method")


Message: TypeAlias = BaseMessage
Sample: TypeAlias = BaseSample | RootSample

TypeSample_co = TypeVar("TypeSample_co", bound=Sample, covariant=True)


def schematize(
    raw_sample: dict[str, Any] | list[dict[str, Any]],
    schema: type[TypeSample_co],
) -> TypeSample_co:
    try:
        if isinstance(raw_sample, dict):
            return schema(**raw_sample)
        elif isinstance(raw_sample, list):
            from data_alchemy.datasets_.generic import GenericSample
            if schema == GenericSample:
                return schema(messages=raw_sample)
            else:
                return schema(root=raw_sample)
        # elif isinstance(raw_sample, (BaseSample, RootSample)):
        elif isinstance(raw_sample, Sample):
            return raw_sample
        else:
            raise TypeError(f"Unsupported type for raw_sample: {type(raw_sample)}")
    except ValidationError as e:
        logger.error(f"Error while schematizing sample: {e}\nGot raw_sample:\n{raw_sample}")
        raise e


IDType: TypeAlias = int


class IDsCollection(OrderedSet[IDType]):
    """A collection of IDs that can be used to filter samples"""


# class BaseDataset(Generic[TypeSample_co], Collection[TypeSample_co]):
class BaseDataset(Collection[TypeSample_co]):
    def __init__(
        self,
        path_or_list_or_meta: str | Path | list[RawSample] | dict[IDType, TypeSample_co],
        ids: Iterable[IDType] | None = None,
        schema: type[TypeSample_co] | None = None,
    ) -> None:
        super().__init__()

        if schema is None:
            raise TypeError("Schema must be provided")
        self.schema: type[TypeSample_co] = schema

        if isinstance(path_or_list_or_meta, (str, Path)):
            self._init_by_path(path_or_list_or_meta, ids)
        elif isinstance(path_or_list_or_meta, list):
            self._init_by_list(path_or_list_or_meta, ids)
        elif isinstance(path_or_list_or_meta, dict):
            self._init_by_meta(path_or_list_or_meta)
        else:
            raise TypeError(
                f"Unsupported type for path_or_list_or_meta: {type(path_or_list_or_meta)}"
            )

    def _init_by_path(
        self,
        path: str | Path,
        ids: Iterable[IDType] | IDsCollection | None = None,
    ) -> None:
        """Initialize the dataset from a path"""
        from data_alchemy.utils.io import load_samples  # pylint: disable=import-outside-toplevel

        samples: list[dict[str, Any] | list[dict[str, Any]]] = load_samples(str(path))
        self._init_by_list(samples, ids)

    def _init_by_list(
        self,
        raw_samples: list[dict[str, Any] | list[dict[str, Any]]],
        ids: Iterable[IDType] | IDsCollection | None = None,
    ) -> None:
        """Initialize the dataset from a list of raw or schematized samples"""
        samples: Sequence[TypeSample_co] = [
            schematize(raw_smp, self.schema)
            for raw_smp in tqdm(raw_samples, desc="Schematizing samples")
        ]

        if ids is None:
            ordered_ids: IDsCollection = IDsCollection(range(len(samples)))
        elif isinstance(ids, IDsCollection):
            ordered_ids: IDsCollection = ids
        elif isinstance(ids, Sequence):
            ordered_ids: IDsCollection = IDsCollection(ids)
        else:
            raise TypeError(f"Unsupported type for ids: {type(ids)}")

        assert len(ordered_ids) == len(samples), "IDs and samples must have the same length"

        self.meta: dict[IDType, TypeSample_co] = dict(zip(ordered_ids, samples))

    def _init_by_meta(self, meta: dict[IDType, TypeSample_co]) -> None:
        """Initialize the dataset from a dictionary of samples"""
        self.meta: dict[IDType, TypeSample_co] = meta

    def save(self, path: str | Path) -> None:
        """Save the dataset to a file"""
        from data_alchemy.utils.io import dump_samples  # pylint: disable=import-outside-toplevel

        dump_samples(self.values(), path)

    def __add__(self, other: "BaseDataset[TypeSample_co]") -> "BaseDataset[TypeSample_co]":
        """Extend this dataset with another dataset"""
        assert self.schema == other.schema, "Cannot concatenate datasets with different schemas"
        offset: int = len(self)
        self.meta.update({(id_ + offset): smp for id_, smp in other.meta.items()})
        return self

    def __contains__(self, ids: object) -> bool:
        if isinstance(ids, IDType):
            return ids in self.meta
        elif isinstance(ids, IDsCollection):
            return all(id_ in self.meta for id_ in ids)
        else:
            raise TypeError(f"Unsupported type for ids: {type(ids)}")

    def __iter__(self) -> Iterator[TypeSample_co]:
        return iter(self.meta.values())

    def __next__(self) -> TypeSample_co:
        return next(iter(self.meta.values()))

    def __len__(self) -> int:
        return len(self.meta)

    def __setitem__(self, key: IDType, role: TypeSample_co) -> None:
        self.meta[key] = role

    def __getitem__(
        self, key: IDType | Iterable[IDType] | IDsCollection
    ) -> TypeSample_co | list[TypeSample_co] | "BaseDataset[TypeSample_co]":
        if isinstance(key, IDType):
            return self.meta[key]
        elif isinstance(key, (Iterable, IDsCollection)):
            return self.select_by_ids(key)
        else:
            raise KeyError(f"Unsupported type for key: {type(key)}")

    def keys(self) -> Iterable[IDType]:
        return self.meta.keys()

    def values(self) -> Iterable[TypeSample_co]:
        return self.meta.values()

    def items(self) -> Iterable[tuple[IDType, TypeSample_co]]:
        return self.meta.items()

    def to_json(self) -> list[dict[str, Any]]:
        return [sample.model_dump(by_alias=True) for sample in self.meta.values()]

    def select_by_ids(self, ids: Iterable[IDType]) -> "BaseDataset[TypeSample_co]":
        try:
            return BaseDataset({id_: self.meta[id_] for id_ in ids}, schema=self.schema)
        except KeyError as e:
            logger.error(f"meta keys: {self.meta.keys()}")
            logger.error(f"ids: {ids}")
            raise KeyError(f"Invalid ID: {e}") from e

    def exclude_by_ids(self, ids: Iterable[IDType]) -> "BaseDataset[TypeSample_co]":
        return BaseDataset(
            # DEBUG: hot fix
            # {id_: smp for id_, smp in tqdm(self.meta.items(), desc="excluding samples") if id_ not in IDsCollection(ids)},
            {id_: smp for id_, smp in self.meta.items() if id_ not in ids},
            schema=self.schema,
        )

    def get_subset(self, n: int) -> "BaseDataset[TypeSample_co]":
        return self.select_by_ids(random.sample(list(self.keys()), n))

    def shuffle(self) -> "BaseDataset[TypeSample_co]":
        "Shuffle the dataset"
        random.shuffle(shuffled_ids := list(self.keys()))
        self.meta = {  # pylint: disable=attribute-defined-outside-init
            id_: self.meta[id_] for id_ in shuffled_ids
        }
        return self

    def sorted(
        self, key: Callable[[Any, Any], bool] | None = None, reverse=False
    ) -> "BaseDataset[TypeSample_co]":
        "Sort the dataset in-place"
        self.meta = dict(  # pylint: disable=attribute-defined-outside-init
            sorted(self.items(), key=key, reverse=reverse)
        )
        return self

    def append(self, sample: TypeSample_co, id_: IDType | None = None):
        if id_ is None:
            id_ = len(self)
        self.meta[id_] = sample

    def extend(self, other: "BaseDataset[TypeSample_co]") -> None:
        return self + other
