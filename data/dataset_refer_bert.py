import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms as T
import random
import clip
import argparse
import h5py
from refer.refer import REFER



class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False,
                 prompt_ensemble=False,
                 coco_instance_gt=False,
                 mask2former=False):
        '''

        :param args: args
        :param image_transforms: get_transforms(args), Resize, ToTensor, Normalize, T.Compose(transforms)
        :param target_transforms: None
        :param split: 'train' or 'val'
        :param eval_mode:
        '''
        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.max_tokens = 20
        self.Cat_dict = self.refer.Cats


        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.sentence_raws = []
        self.cat_names = []

        self.eval_mode = eval_mode

        for r in ref_ids:
            ref = self.refer.Refs[r]

            text_embedding = []
            sentence_raw_for_ref = []
            cat_name = self.Cat_dict[ref['category_id']]
            self.cat_names.append(cat_name)

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']

                if prompt_ensemble:
                    text = [template.format(sentence_raw) for template in ReferDataset.templates]
                    text = clip.tokenize(text)
                else:

                    text = clip.tokenize(sentence_raw)
                text_embedding.append(text)
                sentence_raw_for_ref.append(sentence_raw)

            self.input_ids.append(text_embedding)
            self.sentence_raws.append(sentence_raw_for_ref)

        self.coco_instance_gt = coco_instance_gt
        if self.coco_instance_gt:
            path2img = './refer/data/images/mscoco/images/train2014/'
            path2ann = './data/coco_train_2014_annotation/annotations/instances_train2014.json'

            from pycocotools.coco import COCO
            self.coco = COCO(path2ann)

            self.coco_instance_cat_dict = self.coco.cats

        self.mask2former = mask2former


    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]
        this_cat_name = self.cat_names[index]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])

        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.eval_mode:
            embedding = []
            att = []
            sentence_raw = []
            for s in range(len(self.input_ids[index])):
                text_embedding = self.input_ids[index][s]
                embedding.append(text_embedding.unsqueeze(-1))

            sentence_raw = self.sentence_raws[index]
            tensor_embeddings = torch.cat(embedding, dim=-1)

        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            sentence_raw = self.sentence_raws[index][choice_sent]

        if self.coco_instance_gt:
            coco_instance_target = self.coco.loadAnns(self.coco.getAnnIds(this_img_id))
            BoxAnns = []
            MaskAnns = []
            cat_names = []
            for t in coco_instance_target:
                BoxAnn = t['bbox']
                BoxAnns.append(BoxAnn)

                MaskAnn = self.coco.annToMask(t)
                MaskAnn = T.ToTensor()(MaskAnn)
                MaskAnn = T.Resize((np.asarray(img).shape[0], np.asarray(img).shape[1]))(MaskAnn)
                MaskAnns.append(MaskAnn)

                cat_id = t['category_id']
                cat_name = self.coco_instance_cat_dict[cat_id]['name']
                cat_names.append(cat_name)
            MaskAnns = torch.stack(MaskAnns, dim=1).squeeze(0).type(torch.bool) if len(MaskAnns) != 0 else []

        else:
            BoxAnns = []
            MaskAnns = []
            cat_names = []

        data = []

        if self.mask2former:
            resized_img = np.asarray(img)

        else:
            resized_img = T.Resize(800)(img)
            resized_img = T.ToTensor()(resized_img)
            resized_img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(resized_img)

        data.append(dict(image=resized_img, height=np.asarray(img).shape[0], width=np.asarray(img).shape[1],
                         file_name=this_img['file_name'], cat_name=this_cat_name, img_id=this_img_id,
                         coco_instance_gt=MaskAnns, coco_instance_gt_box=BoxAnns ,coco_instance_cat=cat_names))

        return data, np.asarray(annot), tensor_embeddings, sentence_raw




def get_parser():
    parser = argparse.ArgumentParser(description='Beta model')
    parser.add_argument('--clip_model', default='RN50', help='CLIP model name', choices=['RN50', 'RN101', 'RN50x4', 'RN50x64'])
    parser.add_argument('--visual_proj_path', default='./pretrain/', help='')
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--split', default='val', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')

    return parser


if __name__ == '__main__':
    args = get_parser()
    image_transforms = T.Compose([T.Resize(args.img_size, args.img_size),
                                  T.ToTensor(),
                                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ds = ReferDataset(args,
                      image_transforms=image_transforms,
                      target_transforms=None,
                      eval=True)
