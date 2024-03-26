import os
from typing import Any, Callable, Dict, Optional
import numpy as np
import pydicom
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from torchvision import datapoints

def ngrams(input: list, n):
    outputs = []
    for i in range(len(input) - n + 1):
        outputs.append(input[i : i + n])
    return outputs

class MayoSequenceCTData(Dataset):
    val_name = 'L333'
    test_name = 'L506'
    max_val = 3200
    min_val = -2048
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        seq_len: int =4,
        affine_transforms: Optional[Callable] = None,
        image_transforms: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.affine_transforms = affine_transforms
        self.image_transforms = image_transforms
        self.seq_len = seq_len
        self.split = split
        self.data = self.load_data(split=split)

    def __len__(self):
        return len(self.data)
    
    def read_ima(self, path):
        ds = pydicom.dcmread(path)
        x = ds.pixel_array
        x = pydicom.pixel_data_handlers.util.apply_modality_lut(x, ds)
        x = (x - self.min_val) / (self.max_val - self.min_val)
        
        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        paths = self.data[idx]
        images = []
        for p in paths:
            data = self.read_ima(p)
            images.append(data[None])
        # images = torch.as_tensor(images)[:, None]
        # print(images.shape)
        images = np.asarray(images)
        seq = datapoints.Video(images)
        
        seq = self.affine_transforms(seq)
        seq = self.image_transforms(seq)
        
        return seq

    def load_data(self, split):
        test_im_path_list = []
        val_im_path_list = []
        train_im_path_list = []
        dir_path = os.path.join(
            self.data_dir, "{}_1mm".format("full")
        )
        dir_list = os.listdir(dir_path)
        for dir in dir_list:
            path_list = []
            if dir.find(self.test_name) != -1:
                for root, dirs, files in os.walk(os.path.join(dir_path, dir)):
                    for f in sorted(files):
                        if f.endswith(".IMA"):
                            path_list.append(os.path.join(root, f))
                grams = ngrams(path_list, n=self.seq_len)
                test_im_path_list.extend(grams)
            elif dir.find(self.val_name) != -1:
                for root, dirs, files in os.walk(os.path.join(dir_path, dir)):
                    for f in sorted(files):
                        if f.endswith(".IMA"):
                            path_list.append(os.path.join(root, f))
                grams = ngrams(path_list, n=self.seq_len)
                val_im_path_list.extend(grams)
            else:
                for root, dirs, files in os.walk(os.path.join(dir_path, dir)):
                    for f in sorted(files):
                        if f.endswith(".IMA"):
                            path_list.append(os.path.join(root, f))
                grams = ngrams(path_list, n=self.seq_len)
                train_im_path_list.extend(grams)
                        
        if split == 'train':
            return train_im_path_list
        elif split == 'val':
            return val_im_path_list
        elif split == 'trainval':
            return train_im_path_list + val_im_path_list
        elif split == 'test':
            return test_im_path_list
        elif split == 'all':
            return train_im_path_list + val_im_path_list + test_im_path_list

if __name__ == "__main__":
    data_dir = "/home/ch4090/hlc/project/ct_reconstruction_ddim/data/dose"
    
    image_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x * 2 - 1)]
        )
        
    affine_transforms = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                ) if 256 != 512 else transforms.Lambda(lambda x: x),
                transforms.CenterCrop(256),
            ]
        )
    dataset = MayoSequenceCTData(data_dir, 'all', 4, affine_transforms, image_transforms)
    print(dataset, len(dataset))
    train_dl = DataLoader(dataset, batch_size=4)
    for x in train_dl:
        print(x.shape, x.max(), x.min())
