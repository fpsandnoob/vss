import os
from typing import Any, Callable, Dict, Optional
import pydicom

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class MayoCTData(Dataset):
    val_name = 'L333'
    test_name = 'L506'
    # max_val = 3200
    # min_val = -2048
    
    max_val = 3096
    min_val = -1000
    
    # max_val = 3072
    # min_val = -1024
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        affine_transforms: Optional[Callable] = None,
        image_transforms: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.affine_transforms = affine_transforms
        self.image_transforms = image_transforms
        self.split = split
        self.data = self.load_data(split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.data[idx]
        ds = pydicom.dcmread(x)
        x = ds.pixel_array
        x = pydicom.pixel_data_handlers.util.apply_modality_lut(x, ds)
        x = (x - self.min_val) / (self.max_val - self.min_val)
        
        x = self.affine_transforms(x)
        x = self.image_transforms(x)
        
        return x

    def load_data(self, split):
        test_im_path_list = []
        val_im_path_list = []
        train_im_path_list = []
        for root, dirs, files in os.walk(os.path.join(self.data_dir, "full_1mm")):
            for f in sorted(files):
                if f.endswith(".IMA"):
                    if root.find(self.test_name) != -1:
                        test_im_path_list.append(os.path.join(root, f))
                    elif root.find(self.val_name) != -1:
                        val_im_path_list.append(os.path.join(root, f))
                    else:
                        train_im_path_list.append(os.path.join(root, f))
                        
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
                transforms.ToTensor(),
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                ) if 256 != 512 else transforms.Lambda(lambda x: x),
                transforms.CenterCrop(256),
            ]
        )
    dataset = MayoCTData(data_dir, 'test', affine_transforms, image_transforms)
    print(dataset, len(dataset))
    train_dl = DataLoader(dataset, batch_size=4)
    for x in train_dl:
        print(x.shape, x.max(), x.min())
