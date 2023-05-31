from PIL import Image, ImageDraw
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
from PhraseCutDataset.utils.refvg_loader import RefVGLoader
import torch

class PhraseCutDataset(data.Dataset):
    def __init__(self, split='val', unseen_mode=False, seen_mode=False):
        self.refvg_loader = RefVGLoader(split=split)

        self.COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.unseen_mode = unseen_mode
        self.seen_mode = seen_mode

    def __len__(self):
        return len(self.refvg_loader.img_ids)

    def __getitem__(self, index):
        image_id = self.refvg_loader.img_ids[index]
        img_ref_data = self.refvg_loader.get_img_ref_data(image_id)
        image_id = img_ref_data['image_id']
        image_dir = f'./PhraseCutDataset/data/VGPhraseCut_v0/images/{image_id}.jpg'
        image = Image.open(image_dir).convert('RGB')

        img = T.Resize(800)(image)
        img = T.ToTensor()(img)
        img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        gt_boxes, gt_masks = [], []
        phrases = []
        width, height = img_ref_data['width'], img_ref_data['height']

        cat_count = 0
        for task_i, task_id in enumerate(img_ref_data['task_ids']):

            instances = len(img_ref_data['gt_Polygons'][task_i])
            cat_name = img_ref_data['img_ins_cats'][cat_count]
            cat_count += instances

            if self.unseen_mode and cat_name in self.COCO_CLASSES:
                continue
            elif self.seen_mode and cat_name not in self.COCO_CLASSES:
                continue


            phrases.append(img_ref_data['phrases'][task_i])

            gt_box = img_ref_data['gt_boxes'][task_i] # (x1, y1, x2, y2)

            gt_Mask = img_ref_data['gt_Polygons'][task_i]
            gt_mask = list()

            for ps in gt_Mask:
                gt_mask += ps


            mask = self.polygons_to_mask(gt_mask, img_ref_data['width'], img_ref_data['height'])
            mask = T.ToTensor()(mask)
            gt_masks.append(mask)

            box = self.boxes_region(gt_box)
            gt_boxes.append(box)

        a = torch.ones((1,1))
        if len(gt_masks) == 0:
            return a

        data = []
        data.append(dict(image=img, width=width, height=height, img_id=image_id, gt_masks=gt_masks, phrase=phrases, cat_name=img_ref_data['img_ins_cats']))

        return data


    def boxes_region(self, boxes):
        """
        :return: [x_min, y_min, x_max, y_max] of all boxes
        """

        boxes = np.array(boxes)
        min_xy = np.min(boxes[:, :2], axis=0)
        max_xy = np.max(boxes[:, 2:], axis=0)
        return [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]

    def polygons_to_mask(self, polygons, w, h):
        p_mask = np.zeros((h, w))
        for polygon in polygons:
            if len(polygon) < 2:
                continue
            p = []
            for x, y in polygon:
                p.append((int(x), int(y)))
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(p, outline=1, fill=1)
            mask = np.array(img)
            p_mask += mask
        p_mask = p_mask > 0
        return p_mask