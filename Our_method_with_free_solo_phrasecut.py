import sys
sys.path.append('./')
import argparse
import clip
import torch
import os

Height, Width = 224, 224

from detectron2.checkpoint import DetectionCheckpointer
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import spacy
from tqdm import tqdm
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer

from freesolo.engine.trainer import BaselineTrainer

# hacky way to register
import freesolo.data.datasets.builtin
from freesolo.modeling.solov2 import PseudoSOLOv2

# refer dataset
from data.dataset_refer_bert import ReferDataset
from model.backbone import clip_backbone, CLIPViTFM
from utils import default_argument_parser, setup, Compute_IoU, NMS_for_seg, max_iou_result, extract_noun_phrase
from collections import defaultdict

# phraseCut dataset
from data.dataset_phrasecut import PhraseCutDataset

def main(args, Height, Width):
    assert args.eval_only, 'Only eval_only available!'
    cfg = setup(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    split = 'val'
    unseen_mode = True
    seen_mode = False
    dataset = PhraseCutDataset(split=split, unseen_mode=unseen_mode, seen_mode=seen_mode)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    Trainer = BaselineTrainer
    Free_SOLO = Trainer.build_model(cfg)
    Free_SOLO.eval()

    mode = 'ViT'  # or ViT
    assert (mode == 'Res') or (mode == 'ViT'), 'Specify mode(Res or ViT)'

    Model = clip_backbone(model_name='RN50').to(device) if mode == 'Res' else CLIPViTFM(model_name='ViT-B/32').to(device)

    DetectionCheckpointer(Free_SOLO, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    nlp = spacy.load('en_core_web_lg')

    cum_I, cum_U =0, 0
    mIoU = []

    v = 0.85 if args.dataset == 'refcocog' else 0.95
    r = 0.5


    tbar = tqdm(data_loader)

    for i, data in enumerate(tbar):
        if data == torch.ones((1,1)):
            continue
        image, target, phrases, height, width = data[0]['image'], data[0]['gt_masks'], data[0]['phrase'], data[0]['height'], data[0]['width']

        pred = Free_SOLO(data)[0]

        pred_masks = pred['instances'].pred_masks
        pred_boxes = pred['instances'].pred_boxes

        if len(pred_masks) == 0:
            print('No pred masks')
            continue

        original_imgs = T.Resize((height, width))(image.to(device))
        imgs_for_clip = T.Resize((224, 224))(image.to(device))

        cropped_imgs = []
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)

        for pred_box, pred_mask in zip(pred_boxes.__iter__(), pred_masks):
            pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
            masked_image = original_imgs * pred_mask[None, None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean

            x1, y1, x2, y2 = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
            masked_image = TF.resized_crop(masked_image.squeeze(0), y1, x1, (y2 - y1), (x2 - x1), (Height, Width))
            cropped_imgs.append(masked_image)

        cropped_imgs = torch.stack(cropped_imgs, dim=0)  # [46, 3, 224, 224]

        mask_features = Model.feature_map_masking(imgs_for_clip, pred_masks) if mode == 'Res' else Model(imgs_for_clip, pred_masks, masking_type='token_masking', masking_block=9)

        crop_features = Model.get_gloval_vector(cropped_imgs) if mode == 'Res' else Model(cropped_imgs, pred_masks=None, masking_type='crop')

        ## fixed parameter
        visual_feature = v * mask_features + (1 - v) * crop_features



        for j, phrase in enumerate(phrases):
            phrase = phrase[0].lower()
            doc = nlp(phrase)
            sentence_for_spacy = []

            for i, token in enumerate(doc):
                if token.text == ' ':
                    continue
                sentence_for_spacy.append(token.text)

            sentence_for_spacy = ' '.join(sentence_for_spacy)
            sentence_token = clip.tokenize(sentence_for_spacy).to(device)
            noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(sentence_for_spacy, nlp, need_index=True)
            noun_phrase_token = clip.tokenize(noun_phrase).to(device)

            sentence_features = Model.get_text_feature(sentence_token) if mode == 'Res' else Model.model.encode_text(sentence_token)
            noun_phrase_features = Model.get_text_feature(noun_phrase_token) if mode == 'Res' else Model.model.encode_text(noun_phrase_token)
            text_ensemble = r * sentence_features + (1 - r) * noun_phrase_features

            score = Model.calculate_similarity_score(visual_feature, text_ensemble) if mode == 'Res' else Model.calculate_score(visual_feature, text_ensemble)
            max_index = torch.argmax(score)
            result_seg = pred_masks[max_index]

            _, mIoU, cum_I, cum_U = Compute_IoU(result_seg, target[j].to(device), cum_I, cum_U, mIoU)



    f = open('./result_log/our_method_with_free_solo_phrase.txt', 'a')
    f.write(f'\n\n Model: {mode} / All'
            f'\nDataset: phrase / {split} / unseen mode {"True" if unseen_mode else "False"} / seen mode {"True" if seen_mode else "False"} / Image size = {Height} / {Width}'
            f'\nOverall IoU')

    overall = cum_I * 100.0 / cum_U
    mean_IoU = torch.mean(torch.tensor(mIoU)) * 100.0

    f.write(f'\noverall: {overall:.2f}')
    f.write(f'mIoU: {mean_IoU:.2f}')
    print(f'oIoU: {overall:.2f}')
    print(f'mIoU: {mean_IoU:.2f}')



if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    main(args, Height, Width)
