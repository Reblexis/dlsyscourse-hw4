import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """

        if not os.path.exists(base_folder):
            raise FileNotFoundError(f"{base_folder} not found.")

        with open(f"{base_folder}/batches.meta", "rb") as f:
            self.label_names = pickle.load(f, encoding="bytes")[b"label_names"]

        if train:
            self.X, self.y = np.zeros((50000, 3, 32, 32), dtype=np.float32), np.zeros((50000,), dtype=np.int32)
            for i in range(5):
                with open(f"{base_folder}/data_batch_{i+1}", "rb") as f:
                    data = pickle.load(f, encoding="bytes")
                    self.X[i*10000:(i+1)*10000] = data[b"data"].reshape(-1, 3, 32, 32) / 255.
                    self.y[i*10000:(i+1)*10000] = np.array(data[b"labels"], dtype=np.int32)

        else:
            self.X, self.y = np.zeros((10000, 3, 32, 32), dtype=np.float32), np.zeros((10000,), dtype=np.int32)
            with open(f"{base_folder}/test_batch", "rb") as f:
                data = pickle.load(f, encoding="bytes")
                self.X = data[b"data"].reshape(10000, 3, 32, 32) / 255.
                self.y = np.array(data[b"labels"], dtype=np.int32)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """

        return self.X[index], self.y[index]

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.X.shape[0]
