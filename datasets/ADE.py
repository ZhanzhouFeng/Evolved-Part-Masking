import os
import collections
from torchvision.datasets.vision import VisionDataset
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
import warnings
import cv2
import numpy as np

class _ADEBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        train_set=['training','validation']
        dir_set = ["images", "annotations"]
        file_type=['.jpg','.png']

        ade_root=os.path.join(root,'ade')
        ade_root=os.path.join(ade_root,'ADEChallengeData2016')

        splits_dir = os.path.join(ade_root, dir_set[0],train_set[0])
        file_names=[os.path.basename(file[:-4]) for file in os.listdir(splits_dir)]

        if image_set is 'train':
            image_dir = os.path.join(ade_root, dir_set[0],train_set[0])
            target_dir = os.path.join(ade_root, dir_set[1],train_set[0])
        else:
            image_dir = os.path.join(ade_root, dir_set[0],train_set[1])
            target_dir = os.path.join(ade_root, dir_set[1],train_set[1])


        self.images = [os.path.join(image_dir, x + file_type[0]) for x in file_names]
        self.targets = [os.path.join(target_dir, x + file_type[1]) for x in file_names]

        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)


class ADE20K(_ADEBase):

    @property
    def masks(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        # target = cv2.imread(self.masks[index],cv2.IMREAD_GRAYSCALE)
        # target = Image.fromarray(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target[:,:,0]
