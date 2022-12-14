import os

import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pyad.utilities import math

scaler_map = {
    "minmax": MinMaxScaler,
    "minmaxscaler": MinMaxScaler,
    "standard": StandardScaler,
    "standardscaler": StandardScaler
}


def train_test_split_normal_data(
        X: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        normal_size: float = 1.,
        normal_str_repr: str = "0",
        seed=None,
        shuffle_normal: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split matrix into random train and test subsets.

    X: np.ndarray
        Data matrix that will be split

    y: np.ndarray
        Binary labels (0, 1)

    labels: np.ndarray
        Text representation of the labels

    normal_size: float
        Optional argument to further subsample samples with where y == 0 (normal data)

    normal_str_repr: float
        String representation of the label for normal samples (e.g. "Benign", "normal" or "0")

    seed: int
        Set seeder for reproducibility

    shuffle_normal: bool
        Whether normal data should be shuffled or not (defaults to True)
    """
    msg = "`normal_size` parameter must be inclusively in the range (0, 1], got {:2.4f}"
    assert 0. < normal_size <= 1., msg.format(normal_size)
    if seed:
        np.random.seed(seed)
    # separate normal and abnormal data
    normal_data = X[y == 0]
    abnormal_data = X[y == 1]
    # shuffle normal data
    if shuffle_normal:
        np.random.shuffle(normal_data)
    n_normal = int(((len(normal_data) * normal_size) // 2))
    # train, test split
    if normal_size == 1.:
        # train (normal only)
        X_train = normal_data[:n_normal]
        # test (normal + attacks)
        X_test_normal = normal_data[n_normal:]
    else:
        # train (normal only)
        X_train = normal_data[:n_normal]
        # test (normal + attacks)
        X_test_normal = normal_data[n_normal:n_normal*2]
    X_test = np.concatenate(
        (X_test_normal, abnormal_data)
    )
    y_test = np.concatenate((
        np.zeros(len(X_test_normal), dtype=np.int8),
        np.ones(len(abnormal_data), dtype=np.int8)
    ))
    test_labels = np.concatenate((
        np.array([normal_str_repr] * len(X_test_normal)),
        labels[y == 1]
    ))
    # sanity check: no attack labels associated with 0s and no normal labels associated with 1s
    for bin_y, label in zip(y_test, test_labels):
        assert bin_y == 0 and label == normal_str_repr or bin_y == 1 and label != normal_str_repr
    return X_train, X_test, y_test, test_labels


class SimpleDataset(Dataset):
    def __init__(self, X, y, labels=None):
        self.X = X
        self.y = y
        self.labels = labels
        if self.labels is None:
            self.labels = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index], self.labels[index]


class TabularDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            scaler: str = None,
            normal_size: float = 1.,
            labels_col_name: str = None,
            binary_labels_col_name: str = None,
            normal_str_repr: str = None
    ):
        super(TabularDataset, self).__init__()
        self.data_dir = data_dir
        self.name = data_dir.split(os.path.sep)[-1].split(".")[0].lower()
        self.labels = np.array([])
        self.batch_size = batch_size
        self.normal_size = normal_size
        self.labels_col_name = labels_col_name
        self.binary_labels_col_name = binary_labels_col_name
        if scaler is not None and scaler.lower() != "none":
            assert scaler.lower() in set(scaler_map.keys()), "unknown scaler %s, please use %s" % (
                scaler, scaler_map.keys())
            self.scaler = scaler_map[scaler.lower()]()
        else:
            self.scaler = None
        data = self._load_data(data_dir)
        self.X = data[:, :-1]
        self.y = data[:, -1]
        if self.labels.size == 0:
            self.labels = self.y
        self.shape = self.X.shape
        self.anomaly_ratio = (self.y == 1).sum() / len(self.X)
        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]
        self.normal_str_repr = normal_str_repr or math.get_most_recurrent(self.labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index], self.labels[index]

    def get_params(self) -> dict:
        return {
            "scaler": self.scaler.__class__.__name__,
            "batch_size": self.batch_size,
            "normal_size": self.normal_size,
            "data_dir": self.data_dir
        }

    def _load_data(self, path: str):
        if path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".mat"):
            data = scipy.io.loadmat(path)
            data = np.concatenate((data['X'], data['y']), axis=1)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            labels = df[self.labels_col_name].to_numpy()
            binary_labels = df[self.binary_labels_col_name].to_numpy(dtype=np.int8)
            data = df.drop([self.labels_col_name, self.binary_labels_col_name], axis=1).to_numpy()
            data = np.concatenate((
                data, np.expand_dims(binary_labels, 1)
            ), axis=1)
            self.labels = labels
            assert np.isnan(data).sum() == 0, "detected nan values"
            assert data[data < 0].sum() == 0, "detected negative values"
        else:
            raise RuntimeError(f"Could not open {path}. Dataset can only read .npz and .mat files.")
        return data

    def train_test_split(self, seed=None) -> Tuple[SimpleDataset, SimpleDataset]:
        # train,test split
        X_train, X_test, y_test, test_labels = train_test_split_normal_data(
            self.X, self.y, self.labels,
            seed=seed, normal_str_repr=self.normal_str_repr, normal_size=self.normal_size
        )
        # normalize data
        if self.scaler:
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)

        # convert numpy arrays to dataset objects
        train_set = SimpleDataset(X=X_train, y=np.zeros(len(X_train), dtype=np.int8), labels=np.zeros(len(X_train)))
        test_set = SimpleDataset(X=X_test, y=y_test, labels=test_labels)

        return train_set, test_set

    def loaders(
            self,
            test_size: float = 0.5,
            num_workers: int = 0,
            seed=None,
            shuffle=False
    ) -> (DataLoader, DataLoader):
        train_set, test_set = self.train_test_split(seed)

        # create dataloaders
        train_ldr = DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle
        )
        test_ldr = DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle
        )
        return train_ldr, test_ldr
