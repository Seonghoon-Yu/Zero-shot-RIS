import argparse
import clip
import torch
import os

Height, Width = 224, 224

from detectron2.checkpoint import DetectionCheckpointer
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import spacy
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer
import tqdm

from freesolo.engine.trainer import BaselineTrainer

# hacky way to register
import freesolo.data.datasets.builtin
from freesolo.modeling.solov2 import PseudoSOLOv2

# refer dataset
from data.dataset_refer_bert import ReferDataset
from model.backbone import clip_backbone, CLIPViTFM
from utils import default_argument_parser, setup, Compute_IoU, extract_noun_phrase
from collections import defaultdict

def main(args, Height, Width):
    assert args.eval_only, 'Only eval_only available!'
    cfg = setup(args)

    if args.dataset == 'refcocog':
        args.splitBy = 'umd'  # umd or google in refcocog
    else:
        args.splitBy = 'unc'  # unc in refcoco, refcoco+,

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = ReferDataset(args,
                           image_transforms=None,
                           target_transforms=None,
                           split=args.split,
                           eval_mode=True)

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
    m_IoU = []

    v = 0.85 if args.dataset == 'refcocog' else 0.95
    r = 0.5

    tbar = tqdm.tqdm(data_loader)

    for i, data in enumerate(tbar):
        image, target, clip_embedding, sentence_raw = data
        clip_embedding, target = clip_embedding.squeeze(1).to(device), target.to(device)

        pred = Free_SOLO(image)[0]

        pred_masks = pred['instances'].pred_masks
        pred_boxes = pred['instances'].pred_boxes

        if len(pred_masks) == 0:
            print('No pred masks')
            continue

        original_imgs = torch.stack([T.Resize((height, width))(img.to(pred_masks.device)) for img, height, width in
                                     zip(image[0]['image'], image[0]['height'], image[0]['width'])], dim=0)  # [1, 3, 428, 640]
        resized_imgs = torch.stack([T.Resize((Height, Width))(img.to(pred_masks.device)) for img in image[0]['image']], dim=0)  # [1,3,224,224]

        cropped_imgs = []

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(pred_masks.device)

        for pred_box, pred_mask in zip(pred_boxes.__iter__(), pred_masks):
            pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
            masked_image = original_imgs * pred_mask[None, None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean

            x1, y1, x2, y2 = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
            masked_image = TF.resized_crop(masked_image.squeeze(0), y1, x1, (y2 - y1), (x2 - x1), (Height, Width))
            cropped_imgs.append(masked_image)

        cropped_imgs = torch.stack(cropped_imgs, dim=0)

        mask_features = Model.feature_map_masking(resized_imgs, pred_masks) if mode == 'Res' else Model(resized_imgs, pred_masks, masking_type='token_masking', masking_block=9)

        crop_features = Model.get_gloval_vector(cropped_imgs) if mode == 'Res' else Model(cropped_imgs, pred_masks=None, masking_type='crop')

        visual_feature = v * mask_features + (1 - v) * crop_features

        for sentence, j in zip(sentence_raw, range(clip_embedding.size(-1))):
            sentence = sentence[0].lower()
            doc = nlp(sentence)
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

            text_ensemble = r * sentence_features + (1-r) * noun_phrase_features
            score =  Model.calculate_similarity_score(visual_feature, text_ensemble) if mode == 'Res' else Model.calculate_score(visual_feature, text_ensemble)
            max_index = torch.argmax(score)
            result_seg = pred_masks[max_index]

            _, m_IoU, cum_I, cum_U = Compute_IoU(result_seg, target, cum_I, cum_U, m_IoU)


    f = open('./result_log/our_method_with_free_solo.txt', 'a')
    f.write(f'\n\n CLIP Model: {mode}'
            f'\nDataset: {args.dataset} / {args.split} / {args.splitBy}'
            f'\nOverall IoU / mean IoU')

    overall = cum_I * 100.0 / cum_U
    mean_IoU = torch.mean(torch.tensor(m_IoU)) * 100.0

    f.write(f'\n{overall:.2f} / {mean_IoU:.2f}')






if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    opts = ['OUTPUT_DIR', 'training_dir/FreeSOLO_pl', 'MODEL.WEIGHTS', 'checkpoints/FreeSOLO_R101_30k_pl.pth']
    args.opts = opts
    print(args.opts)
    main(args, Height, Width)
