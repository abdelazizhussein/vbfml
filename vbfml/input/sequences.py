from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence, to_categorical

from vbfml.input.uproot import UprootReaderMultiFile


@dataclass
class DatasetInfo:
    name: str
    files: list
    n_events: int
    treename: str
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.name


class MultiDatasetSequence(Sequence):
    def __init__(self, batch_size: int, branches: "list[str]", shuffle=True) -> None:
        self.datasets = {}
        self.readers = {}
        self.branches = branches
        self.batch_size = batch_size
        self._shuffle = shuffle
        self.encoder = LabelEncoder()

    def __len__(self) -> int:
        return self.total_events() // self.batch_size

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool) -> None:
        self._shuffle = shuffle

    def _read_dataframes_for_batch(self, idx: int) -> list:
        """Reads and returns data for a given batch and all datasets"""
        dataframes = []
        for name in self.datasets.keys():
            df = self._read_single_dataframe_for_batch_(idx, name)
            dataframes.append(df)
        return dataframes

    def _get_start_stop_for_single_read(self, idx: int, dataset_name: str) -> tuple:
        """Returns the start and stop coordinates for reading a given batch of data from one dataset"""
        start = np.floor(idx * self.batch_size * self.fractions[dataset_name])
        stop = np.floor((idx + 1) * self.batch_size * self.fractions[dataset_name]) - 1
        return start, stop

    def _read_single_dataframe_for_batch_(self, idx: int, dataset_name: str):
        """Reads and returns data for a given batch and single data"""
        start, stop = self._get_start_stop_for_single_read(idx, dataset_name)
        if not dataset_name in self.readers:
            self._initialize_reader(dataset_name)
        df = self.readers[dataset_name].read_events(start, stop)
        df["label"] = self.encode_label(self.datasets[dataset_name].label)
        return df

    def dataset_labels(self):
        return [dataset.label for dataset in self.datasets.values()]

    def encode_label(self, label):
        return self.label_encoding[label]

    def __getitem__(self, idx: int) -> tuple:
        """Returns a single batch of data"""
        dataframes = self._read_dataframes_for_batch(idx)

        df = pd.concat(dataframes)

        if self.shuffle:
            df = df.sample(frac=1)

        features = df.drop(columns="label").to_numpy()
        labels = np.array(df["label"]).reshape((len(df["label"]), 1))

        return (features, to_categorical(labels, num_classes=len(self.dataset_labels())))

    def total_events(self) -> int:
        """Total number of events of all data sets"""
        return sum(dataset.n_events for dataset in self.datasets.values())

    def _init_dataset_encoding(self) -> None:
        labels = sorted([dataset.label for dataset in self.datasets.values()])
        label_encoding = dict(enumerate(labels))
        label_encoding.update({v: k for k, v in label_encoding.items()})
        self.label_encoding = label_encoding

    def _datasets_update(self):
        """Perform all updates needed after a change in data sets"""
        self._calculate_fractions()
        self._init_dataset_encoding()

    def add_dataset(self, dataset: DatasetInfo) -> None:
        """Add a new data set to the Sequence."""
        if dataset.name in self.datasets:
            raise IndexError(f"Dataset already exists: '{dataset.name}'.")
        self.datasets[dataset.name] = dataset
        self._datasets_update()

    def remove_dataset(self, dataset_name: str) -> DatasetInfo:
        """Remove dataset from this Sequence and return its DatasetInfo object"""
        info = self.datasets.pop(dataset_name)
        self._datasets_update()
        return info

    def get_dataset(self, dataset_name: str) -> DatasetInfo:
        return self.datasets[dataset_name]

    def _initialize_reader(self, dataset_name) -> None:
        """
        Initializes file readers for a given data set.

        Note that this operation may be slow as the reader
        will open all files associated to it to determine
        event counts.
        """
        info = self.datasets[dataset_name]
        reader = UprootReaderMultiFile(
            files=info.files,
            branches=self.branches,
            treename=info.treename,
        )
        self.readers[dataset_name] = reader

    def _initialize_readers(self) -> None:
        """Initializes file readers for all data sets"""
        for dataset_name in self.datasets.keys():
            self._initialize_reader(dataset_name)

    def _calculate_fractions(self) -> None:
        """Determine what fraction of the total events is from a given data set"""
        total = self.total_events()
        self.fractions = {
            name: info.n_events / total for name, info in self.datasets.items()
        }
