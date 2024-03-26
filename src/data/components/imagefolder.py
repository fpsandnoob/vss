from typing import Any, Callable, List, Optional, Union
from torchvision.datasets import ImageFolder, LSUN


class ImageFolderWithoutClass(ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample


if __name__ == "__main__":
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt

    data_dir = "/home/ch4090/hlc/project/dlir/data/celeba_eval"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolderWithoutClass(data_dir, transform=transform)
    print(len(dataset))
    print(dataset[0].shape)
    plt.imshow(dataset[0][0])
    plt.show()


class LSUNRaW(LSUN):
    def __init__(
        self,
        root: str,
        classes: Union[str, List[str]] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, classes, transform, target_transform)

    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index)[0]