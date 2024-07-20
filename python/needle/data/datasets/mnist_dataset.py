from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """

    project_path = ""
    with gzip.open(f"{project_path}{image_filename}", 'rb') as f:
        magic_number = f.read(4)
        items_num = struct.unpack('>I', f.read(4))[0]
        rows_num = struct.unpack('>I', f.read(4))[0]
        columns_num = struct.unpack('>I', f.read(4))[0]

        X = np.zeros((items_num, rows_num * columns_num), dtype=np.float32)
        for i in range(items_num):
            pixels = f.read(rows_num * columns_num)
            image = np.frombuffer(pixels, dtype=np.uint8)
            image_normalized = image.astype(np.float32) / 255
            X[i] = image_normalized

    with gzip.open(f"{project_path}{label_filename}", 'rb') as f:
        magic_number = f.read(4)
        items_num = struct.unpack('>I', f.read(4))[0]
        y = np.zeros(items_num, dtype=np.uint8)
        for i in range(items_num):
            label_byte = f.read(1)
            y[i] = np.frombuffer(label_byte, dtype=np.uint8)

    return X, y


class MNISTDataset(Dataset):
    def __init__(
            self,
            image_filename: str,
            label_filename: str,
            transforms: Optional[List] = None,
    ):
        self.X, self.y = parse_mnist(image_filename, label_filename)
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        x = self.X[index].reshape(-1, 28, 28, 1)
        y = self.y[index]
        for i in range(x.shape[0]):
            x[i] = self.apply_transforms(x[i])

        return x, y

    def __len__(self) -> int:
        return len(self.X)
