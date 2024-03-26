import json
import os
from typing import Any, Callable, Dict, Optional
import pydicom
import tifffile
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


def load_tiff_stack_with_metadata(file):
    """

    :param file: Path object describing the location of the file
    :return: a numpy array of the volume, a dict with the metadata
    """
    if not (file.name.endswith(".tif") or file.name.endswith(".tiff")):
        raise FileNotFoundError("File has to be tif.")
    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value
    metadata = metadata.replace("'", '"')
    try:
        metadata = json.loads(metadata)
    except:
        print("The tiff file you try to open does not seem to have metadata attached.")
        metadata = None
    return data, metadata


class MayoRealProjectionData(Dataset):
    # val_name = 'L333'
    # test_name = 'L506'
    max_val = 3200
    min_val = -2048

    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        voxel_size: float = 0.7,
        fbp_filter: str = "hann",
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.fbp_filter = fbp_filter
        # self.vox_scaling = 1 / self.voxel_size

        self.projections, self.metadata = self.load_data(Path(data_dir))

    def __len__(self):
        return self.projections.shape[-1]

    def __getitem__(self, idx: int):
        prj = np.copy(np.flip(self.projections[:, :, 27], axis=1))

        return (
            prj,
            {
                "image_size": self.image_size,
                "voxel_size": self.voxel_size,
                "fbp_filter": self.fbp_filter,
                "metadata": self.metadata
            },
        )

    def load_data(self, path):
        projections, metadata = load_tiff_stack_with_metadata(path)

        return projections, metadata


if __name__ == "__main__":
    data_dir_ = "/home/ch4090/hlc/project/dlir/data/fanbeam/scan_001_flat_fan_projections.tif"

    dataset = MayoRealProjectionData(data_dir_)
    print(dataset, len(dataset))
    train_dl = DataLoader(dataset, batch_size=4)
    for x, metadata in train_dl:
        print(x.shape, x.max(), x.min())
