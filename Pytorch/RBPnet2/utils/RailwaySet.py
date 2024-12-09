import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RailwaySet(Dataset):
    """RailwaySet Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``railway-set`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    base_folder = "images_cifar"
    data_list = [
        ["data_batch_1"],
        ["test_batch"]
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data: Any = []
        self.targets = []

        # Load the pickled numpy arrays
        if self.train:
            data_list = self.data_list[:1]  # Use data_batch_1 for training
        else:
            data_list = self.data_list[1:]  # Use test_batch for testing

        for file_name in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name[0])
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                print(f"Loaded entry keys: {entry.keys()}")  # Debug print to check the structure of entry

                # Decode keys as UTF-8
                if b"data" in entry:
                    self.data.append(entry[b"data"])
                else:
                    raise KeyError(f"Expected key 'data' not found in {file_path}")

                if b"labels" in entry:
                    self.targets.extend(entry[b"labels"])
                elif b"fine_labels" in entry:
                    self.targets.extend(entry[b"fine_labels"])
                else:
                    raise KeyError(f"Expected key 'labels' or 'fine_labels' not found in {file_path}")

        self.data = np.vstack(self.data).reshape(-1, 3, 224, 224)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            if self.meta["key"] in data:
                self.classes = data[self.meta["key"]]
            else:
                raise KeyError(f"Expected key '{self.meta['key']}' not found in {path}")
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
