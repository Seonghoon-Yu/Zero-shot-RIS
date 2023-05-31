import numpy as np
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as T
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List
import torch
from collections import defaultdict

class CocoDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.image_transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.target_transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224))
        ])

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        # target dict key: ['segmentation', 'area', 'iscrowd', 'image_id, 'bbox', 'category_id', 'id']


        MaskAnns = []
        for t in target:
            MaskAnn = self.coco.annToMask(t)
            MaskAnn = self.target_transform(MaskAnn)
            MaskAnns.append(MaskAnn)
        if len(MaskAnns) != 0:
            MaskAnns = torch.stack(MaskAnns, dim=1).squeeze(0).type(torch.bool) # [20, 224, 224]
        else:
            MaskAnns = []


        image = self.image_transform(image)

        GT = defaultdict(list)
        for i, t in enumerate(target): # target is list, len=20
            Cat_Id = t['category_id']
            GT[Cat_Id].append(MaskAnns[i])


        # target = {'MaskAnns':MaskAnns, 'CatIds': Cat_Ids}


        return image, GT

    def __len__(self) -> int:
        return len(self.ids)

    def annToMask(self, ann):
        mask = self.coco.annToMask()
        return mask

    def IdToCat(self, ids=[]):
        cat = self.coco.loadCats(ids)
        return cat