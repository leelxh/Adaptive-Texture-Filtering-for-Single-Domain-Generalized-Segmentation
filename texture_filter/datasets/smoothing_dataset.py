import math
import os.path
from os.path import join
import random
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, AnyStr, Any

from torchvision.transforms import transforms

from .image_folder import is_image_file, make_dataset
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor()])



class GTA5Dataset(Dataset):
    def __init__(self, root, size=None, is_round=True, downsample=1, ext="png", resize=None):
        super(GTA5Dataset, self).__init__()
        self.root = root
        self.origin_path = sorted(make_dataset(self.root + "/images/"))
        self.origin_size = len(self.origin_path)
        self.round = is_round
        self.patch_size = 256
        self.resize = resize
        if size is None:
            self.size = len(self.origin_path)
        else:
            self.size = min(size, len(self.origin_path))

    def __len__(self) -> int:
        return self.size

    def __getitem__(
            self,
            index: int
    ) -> Dict[AnyStr, Any]:
        if index > self.size:
            raise IndexError("out of the range, indexed %d-th image, but have %d images in total." % (index, self.size))
        paths = self.origin_path[index]
        org_input = Image.open(paths).convert("RGB").resize((640, 480))
        edge_input = Image.open(paths.replace("images", "labels")).convert("RGB").resize((640, 480)) #.resize((256, 256))
        if self.resize is not None : org_input, edge_input = org_input.resize(self.resize), edge_input.resize(self.resize) 
        return  to_tensor(org_input),  \
                to_tensor(edge_input), \
                "".join(self.origin_path[index % self.origin_size].split('/')[-1].split(".")[:-1])