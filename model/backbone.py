import torch
import torch.nn as nn
import clip
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import copy
from einops import rearrange


class clip_backbone(nn.Module):
    ''' CLIP backbone before attention pooling.'''

    def __init__(self, model_name = 'RN50', visual_projs_path = './pretrain/'):
        '''
        Args:
            model_name: availabe models = ['RN50', 'RN101', 'RN50x4', 'RN50x64']
            visual_projs_path = path to 'clip_weight.pth'
        '''
        super().__init__()

        self.model, _ = clip.load(model_name)

        self.visual_projs_path = visual_projs_path + model_name + '_clip_weights.pth'

        self.in_channels = self.model.visual._inplanes # visual feature dimension
        self.text_channels = self.model.visual.output_dim # text feature dimension

        self.v_proj = nn.Conv2d(self.in_channels, self.in_channels, 1).to(self.device)
        self.c_proj = nn.Conv2d(self.in_channels, self.text_channels, 1).to(self.device)

        v_proj_weight = nn.parameter.Parameter(self.model.visual.attnpool.v_proj.weight[:,:,None,None])
        c_proj_weight = nn.parameter.Parameter(self.model.visual.attnpool.c_proj.weight[:,:,None,None])

        self.v_proj.weight = v_proj_weight
        self.c_proj.weight = c_proj_weight

        # self.load_visual_projs()

        self.activation_map = None
        self.activation_map_gradients = None

        # self.text_encoder_attn_mask = self.model.build_attention_mask()

    def load_visual_projs(self):
        print('load visual projs')
        loaded = torch.load(self.visual_projs_path, map_location='cuda')

        for attr in ['v_proj', 'c_proj']:
            current_attr = getattr(self, attr) # self.v_proj
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:,:,None,None]
            current_attr.load_state_dict(state_dict)

    @property
    def device(self):
        return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype


    def forward(self, data, text):
        image = data['img'][0] # [1,3,480,480]
        H,W,_ = data['img_metas'][0][0]['ori_shape']


        x = self.model.encode_image(image.type(self.dtype))
        h, w = x.shape[2], x.shape[3]
        v = self.v_proj(x)

        image_features = self.c_proj(v) # [4, 1024, 15, 15]
        text_features = self.model.encode_text(text).unsqueeze(-1) # [4, 1024, 1]


        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True) # [4(batch), 1024(channel), 15(H), 15(W)]
        text_features = text_features / text_features.norm(dim=1, keepdim=True) # [4(batch), 1024(channel), 1]

        # # upsaple clip's feature map into original image shape
        # image_features = F.interpolate(image_features, size=(H,W), align_corners=False, mode='bilinear')


        ## for score map
        image_features = rearrange(image_features, 'b c h w -> b (h w) c')


        score_map = rearrange(torch.einsum('bij,bjk->bki', image_features, text_features), 'b s (h w) -> b s h w', h=h, w=w)
        # score_map = score_map / score_map.norm(dim=1, keepdim=True)

        # Upsample
        score_map = F.interpolate(score_map, size=(H,W), align_corners=False, mode='bilinear')

        image_features = rearrange(image_features, 'b (h w) c -> b c h w', h=h, w=w)

        return score_map, image_features, text_features

    def get_image_feature(self, data, free_solo=True, size=None):

        # For Free_SOLO(Detectron2) Input
        if free_solo:
            image = data.to(self.device)

            if size:
                self.H, self.W = size
            # self.H, self.W = image.shape[2], image.shape[3] # shorter size = 800

            x = self.model.encode_image(image.type(self.dtype))
            v = self.v_proj(x)

            image_features = self.c_proj(v)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            return image_features

        # For MMdetection Input
        image = data['img'][0] # [1,3,H,W]
        self.H, self.W,_ = data['img_metas'][0][0]['ori_shape']

        x = self.model.encode_image(image.type(self.dtype))
        v = self.v_proj(x)

        image_features = self.c_proj(v) # [batch, text_channel, H, W]

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    def get_text_feature(self, text, target_noun_index=None):
        text_features = self.model.encode_text(text, target_noun_index) #[batch, channel]

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    def text_masking_feature(self, text, masking_index=[], masking_block=11):
        text_encoder = self.model.transformer
        masking_index = [i+1 for i in masking_index] # because start token

        # sentence = text[0]
        # noun_phrase = text[1]


        # text = [1, 77]
        x = self.model.token_embedding(text).type(self.dtype) # [1,77,512]
        x = x + self.model.positional_embedding.type(self.dtype) # [1,77,512]
        x = x.permute(1, 0, 2) # [77, 1, 512]

        for block_idx, resblock in enumerate(text_encoder.resblocks): # last block idx [11, 11, 23]
            if block_idx >= masking_block:
                if masking_index:
                    x[masking_index] = 0
                    x = resblock(x)

                else:
                    x = resblock(x)
            else:
                x = resblock(x)


        x = x.permute(1, 0, 2) # [1, 77, 512]
        x = self.model.ln_final(x).type(self.dtype) # [1, 77, 512]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # [1, 512]
        x = x / x.norm(dim=1, keepdim=True)

        return x



    def get_ensembled_text_feature(self, text):
        # text = [80, 77]
        text_features = self.model.encode_text(text) # text_features = [80,1024]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.mean(0)
        text_features = text_features / text_features.norm() # text_features = [1024] -> [1,1024,1]로 바꿔줘야 한다.

        return text_features[None,:,None] # [1,1024,1]

    def get_score_map(self, image_features, text_features):
        h, w = image_features.shape[2], image_features.shape[3]

        image_features = rearrange(image_features, 'b c h w -> b (h w) c')
        score_map = rearrange(torch.einsum('bij,bjk->bki', image_features, text_features), 'b s (h w) -> b s h w', h=h, w=w)

        # Upsample
        score_map = F.interpolate(score_map, size=(self.H,self.W), align_corners=False, mode='bilinear')

        return score_map

    def get_gloval_vector(self, data, attn_mask=None, free_solo=True):
        image = data.to(self.device)
        if free_solo:
            image_features = self.model.encode_image(image.type(self.dtype), attn=True, attn_mask=attn_mask)

            # normalize
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            return image_features # [1, 1024]

    def project_seg_attn(self, data, pred_masks):
        image, pred_masks = data.to(self.device), pred_masks.to(self.device)

        image_features = self.model.encode_image(image.type(self.dtype), attn=False) # [1,2048,7,7]
        image_features = image_features / image_features.norm(dim=1, keepdim=True) # normalize

        H, W = image_features.shape[2], image_features.shape[3]

        pred_masks = TF.resize(pred_masks, (H,W)) # [N,7,7]

        mask_features = []

        mean_position = torch.ones(1, dtype=torch.bool).to(self.device)
        for pred_mask in pred_masks:
            # attn_mask should be [HW+1, HW+1] and dtype=torch.bool
            attn_mask = pred_mask.reshape(pred_mask.shape[0] * pred_mask.shape[1]) # [49]
            attn_mask = torch.cat([mean_position, attn_mask],dim=0).repeat(pred_mask.shape[0] * pred_mask.shape[1]+1,1) # [50,50]

            attn_mask_feature = self.model.visual.attnpool(image_features, attn_mask=~attn_mask)
            # attn_mask_feature = attn_mask_feature / attn_mask_feature.norm(dim=1, keepdim=True)

            mask_features.append(attn_mask_feature)

        mask_features = torch.stack(mask_features, dim=0) # [N,1,1024]

        return mask_features

    def generate_score_map(self, image, text_features, H,W):
        image_features = self.model.encode_image(image.type(self.dtype), attn=False) # [1,2048,7,7]
        image_features = self.v_proj(image_features) # [1, 2048, 7, 7]
        image_features = self.c_proj(image_features) # [1, 1024, 7, 7]

        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        h, w = image_features.shape[2], image_features.shape[3]

        image_features = rearrange(image_features, 'b c h w -> b (h w) c') # [1,49,1024]
        # text features = [1, 1024]
        score_map = torch.einsum('bij,bjk->bki', image_features, text_features[:,:,None]) # [1,1,49]
        score_map = rearrange(score_map, 'b s (h w) -> b s h w', h=h, w=w) # [1, 1, 7, 7]
        score_map = F.interpolate(score_map, size=(H,W), align_corners=False, mode='bilinear') # [1, 1, 224, 224]
        score_map = (score_map - torch.min(score_map)) / (torch.max(score_map) - torch.min(score_map))

        return score_map


    def feature_map_masking(self, data, pred_masks, query_masking=True, ignore_zero=False,
                            before_normalize=True, after_normalize=False, parallel=True, use_attn_mask=False):
        image, pred_masks = data.to(self.device), pred_masks.to(self.device)

        image_features = self.model.encode_image(image.type(self.dtype), attn=False) # [1,2048,7,7]
        if before_normalize:
            image_features = image_features / image_features.norm(dim=1, keepdim=True) # normalize

        H, W = image_features.shape[2], image_features.shape[3]

        # pred_masks = TF.resize(pred_masks, (H,W)) # [N,7,7]
        pred_masks = TF.resize(pred_masks.type(torch.float32), (H, W))  # [N,7,7]


        if parallel:
            masked_feature_map = torch.mul(image_features, pred_masks[:, None, :, :])

            if query_masking:
                masked_features = self.model.visual.attnpool(masked_feature_map, ignore_zero=ignore_zero)
            else:
                masked_features = self.model.visual.attnpool(masked_feature_map, image_features)

            if after_normalize:
                masked_features = masked_features / masked_features.norm(dim=1, keepdim=True)


        else:
            masked_features = []
            mean_position = torch.ones(1, dtype=torch.bool).to(self.device)
            # masking feature map
            for pred_mask in pred_masks: # [N,7,7], [1,2048,7,7]
                masked_feature_map = torch.mul(image_features, pred_mask[None, None, ...])

                if use_attn_mask:
                    attn_mask = pred_mask.reshape(pred_mask.shape[0] * pred_mask.shape[1])  # [49]
                    attn_mask = torch.cat([mean_position, attn_mask], dim=0).repeat(pred_mask.shape[0] * pred_mask.shape[1] + 1, 1)  # [50,50]
                    masked_feature = self.model.visual.attnpool(masked_feature_map, ignore_zero=ignore_zero, attn_mask=~attn_mask)
                else:
                    masked_feature = self.model.visual.attnpool(masked_feature_map, ignore_zero=ignore_zero)

                if after_normalize:
                    masked_feature = masked_feature / masked_feature.norm(dim=1, keepdim=True)

                masked_features.append(masked_feature)

            masked_features = torch.stack(masked_features, dim=0)

        return masked_features

    def calculate_similarity_score(self, image_features, text_features):
        # image_features = [N, 1,1024]
        # text_feature = [1,1024]
        # logit_scale.exp() = 100. ?? 이거는 학습에 사용되는 건데 inference시에도 곱해줄 필요가 있을까?

        # text_features = text_features.squeeze(-1)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()

        # logits_per_image = logit_scale * image_features @ text_features.t() # 16.5868

        logits_per_image = logit_scale * image_features @ text_features.t() # 0.1659
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image # [N, 1, 1]


    def generate_grad_cam(self, image, noun_text, H=224, W=224, Tokenize=True):
        # noun_text에 대한 tokenize 이후 embedding을 입력 받아야 한다.
        # noun_text에 a photo of a 붙여주기
        image, noun_text = image.to(self.device), noun_text.to(self.device)

        if Tokenize:
            noun_text_feature = self.model.encode_text(noun_text) # [batch, channel]
            noun_text_feature = noun_text_feature / noun_text_feature.norm(dim=1, keepdim=True)
        else:
            noun_text_feature = noun_text

        # Get global vector and image_feature_map
        image_feature_map = self.model.encode_image(image.type(self.dtype), attn=False)
        image_feature_map.register_hook(self.save_gradients)
        global_vector = self.model.visual.attnpool(image_feature_map)
        global_vector = global_vector / global_vector.norm(dim=1, keepdim=True) # this is for backward function

        image_feature_map = image_feature_map / image_feature_map.norm(dim=1, keepdim=True) # For Grad-CAM and CAM

        # Get gradients of feature map
        logit_scale = self.model.logit_scale.exp()
        logit_per_image = logit_scale * global_vector @ noun_text_feature.t()
        logit_per_image *= -1

        self.zero_grad()
        logit_per_image.backward(retain_graph=True)

        # Get Grad-CAM
        gradients_map = self.activation_map_gradients
        weights = torch.mean(gradients_map, axis=(2,3))
        weighted_activations = weights[:,:,None,None] * image_feature_map
        Grad_CAM = weighted_activations.sum(axis=1)


        Grad_CAM = F.interpolate(Grad_CAM[None,:,:,:], size=(H,W), align_corners=False, mode='bilinear')


        return Grad_CAM

    def save_gradients(self, grad):
        self.activation_map_gradients = grad

    def save_activation_map(self, input):
        self.activation_map = input

    def get_activation_map(self):
        return self.activation_map

    def get_gradients(self):
        return self.activation_map_gradients


class clip_vit(nn.Module):
    def __init__(self, model_name='ViT-B/16'):
        super().__init__()
        '''
        model_neam: 'RN50', 'RN101', 'RN50x4', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT0L/14@336px'
        '''


        self.model, _ = clip.load(model_name)

    @property
    def device(self):
        return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    def get_image_feature(self, data):
        image = data.to(self.device)

        image_feature = self.model.encode_image(image) # [1, 50, 512]

        # normalize
        image_feature = image_feature / image_feature.norm(dim=2, keepdim=True)

        image_feature = image_feature[:, 1:, :]
        h, w = int(np.sqrt(image_feature.shape[1])), int(np.sqrt(image_feature.shape[1]))

        image_feature = image_feature.reshape(image_feature.shape[0], h, w, image_feature.shape[2])
        image_feature = image_feature.permute(0,3,1,2)

        return image_feature



    def get_text_feature(self, text):
        text_features = self.model.encode_text(text).unsqueeze(-1) #[batch, channel, 1]

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features



    def get_score_map(self, image_features, text_features):
        h, w = image_features.shape[2], image_features.shape[3]

        image_features = rearrange(image_features, 'b c h w -> b (h w) c')

        score_map = rearrange(torch.einsum('bij,bjk->bki', image_features, text_features), 'b s (h w) -> b s h w', h=h, w=w)

        # # Upsample
        # score_map = F.interpolate(score_map, size=(self.h,self.w), align_corners=False, mode='bilinear')

        return score_map

    def befor_masking(self, data, pred_masks, pixel_mean):
        image = data.to(self.device)

        h, w = image.shape[2], image.shape[3]
        pred_masks = TF.resize(pred_masks, (h, w)).type(torch.float32)

        image_features = []

        masked_img = []
        for pred_mask in pred_masks:
            img = image * pred_mask[None, None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean
            masked_img.append(img)
        masked_img = torch.stack(masked_img, dim=0).squeeze(1)



        image_features = self.get_image_feature(masked_img).squeeze(0)

        return image_features

    def after_masking(self, data, pred_masks):
        image_features = self.forward(data)

        H, W = image_features.shape[2], image_features.shape[3]

        pred_masks = TF.resize(pred_masks, (H,W)) # [N,7,7]

        for pred_mask in pred_masks:
            masked_feature_map = torch.mul(image_features, pred_mask[None, None, ...])
            mean_feature = masked_feature_map.mean()
            print(mean_feature)
            print(mean_feature.shape)


        return image_features

class CLIPViTFM(nn.Module):
    def __init__(self, model_name='ViT-B/32', size=224):
        super().__init__()

        if model_name == 'ViT-B/32':
            self.last_layer = 11
            self.num_heads = 12
        elif model_name == 'ViT-B/16':
            self.last_layer = 11
            self.num_heads = 12
        elif model_name == 'ViT-L/14':
            self.last_layer = 23
            self.num_heads = 16

        self.model, _ = clip.load(model_name)


    @property
    def device(self):
        return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    def text_masking_feature(self, text, masking_index=[], masking_block=11):
        text_encoder = self.model.transformer
        masking_index = [i+1 for i in masking_index] # because start token


        x = self.model.token_embedding(text).type(self.dtype) # [1,77,512]
        x = x + self.model.positional_embedding.type(self.dtype) # [1,77,512]
        x = x.permute(1, 0, 2) # [77, 1, 512]

        for block_idx, resblock in enumerate(text_encoder.resblocks): # last block idx [11, 11, 23]
            if block_idx >= masking_block:
                if masking_index:
                    x[masking_index] = 0
                    x = resblock(x)

                else:
                    x = resblock(x)
            else:
                x = resblock(x)


        x = x.permute(1, 0, 2) # [1, 77, 512]
        x = self.model.ln_final(x).type(self.dtype) # [1, 77, 512]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # [1, 512]

        return x
    #
    # def noun_phrase_masking(self, text, noun_phrase_index=[], masking_block=11, pad_masking=True):
    #     text_encoder = self.model.transformer
    #
    #     if noun_phrase_index:
    #         if pad_masking:
    #             noun_phrase_index = [i + 1 for i in noun_phrase_index]
    #             masking_index = [i not in noun_phrase_index for i in range(77)]
    #         else:
    #             # False 를 77개 만들고, sentence 부분 True로 바꾸고, noun phrase부분을 다시 True
    #             masking_index = [False for i in range(77)]
    #
    #             noun_phrase_index = [i + 1 for i in noun_phrase_index]
    #             eot_token_index = text.argmax(dim=-1)
    #             sentence_index = [i for i in range(1, eot_token_index)]
    #
    #             slice_index = []
    #             for i in sentence_index:
    #                 if i not in noun_phrase_index:
    #                     slice_index.append(i)
    #
    #             for i in slice_index:
    #                 masking_index[i] = True
    #
    #     x = self.model.token_embedding(text).type(self.dtype) # [1,77,512]
    #     x = x + self.model.positional_embedding.type(self.dtype) # [1,77,512]
    #     x = x.permute(1, 0, 2) # [77, 1, 512]
    #
    #     for block_idx, resblock in enumerate(text_encoder.resblocks): # last block idx [11, 11, 23]
    #         if block_idx >= masking_block:
    #             if noun_phrase_index:
    #                 x[masking_index] = 0
    #                 x = resblock(x)
    #             else:
    #                 x = resblock(x)
    #
    #         else:
    #             x = resblock(x)
    #
    #     x = x.permute(1, 0, 2) # [1, 77, 512]
    #     x = self.model.ln_final(x).type(self.dtype) # [1, 77, 512]
    #     x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # [1, 512]
    #
    #     return x



    def calculate_score(self, image_features, text_features, visual_norm_dim=1):
        # image_features = [N,512]
        # text_feature = [1,512]
        # logit_scale.exp() = 100.

        image_features = image_features / image_features.norm(dim=visual_norm_dim, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t() # 0.1659

        return logits_per_image # [N, 1]

    def upsample_pos_emb(self, emb):
        # upsample the pretrained embedding for higher resolution
        # emb size NxD
        first = emb[:1, :]
        emb = emb[1:, :]
        N, D = emb.size(0), emb.size(1)
        n = int(np.sqrt(N))
        assert n * n == N

        emb = emb.permute(1, 0)
        emb = emb.view(1, D, n, n).contiguous()
        emb = F.upsample(emb, size=(self.size), mode='bilinear',
                         align_corners=None)
        emb = emb.view(D, -1).contiguous()
        emb = emb.permute(1, 0)
        emb = torch.cat([first, emb], 0)
        emb = nn.parameter.Parameter(emb.half())
        return emb

    def make_attn_mask(self, pred_masks, size=None):
        # pred_masks = [46, H, W], Torch.bool
        if size is not None:
            pred_masks = TF.resize(pred_masks, size=(size, size))  # [46,7,7]

        cls = torch.ones((1,), dtype=torch.bool).to(self.device)

        N = pred_masks.size(0)
        attn_masks = pred_masks.view(N, -1) # [46, 49]
        attn_masks = [attn_mask.expand(self.num_heads, -1) for attn_mask in attn_masks] # [N, num_heads, L]
        attn_masks = torch.stack(attn_masks, dim=0).view(N * self.num_heads, -1).contiguous() # [N*num_heads, L]
        attn_masks = torch.cat([cls.expand(N * self.num_heads, -1), attn_masks], dim=1) # [N*num_heads, L+1]
        attn_masks = attn_masks.unsqueeze(-1).expand(-1, -1, attn_masks.shape[1]) # [N*num_heads, L+1, L+1]

        return ~attn_masks



    def forward(self, image, pred_masks,  masking_block=None, masking_type='token_masking'):
        if masking_block is None:
            masking_block = self.last_layer

        vit = self.model.visual
        x = image.type(self.model.dtype)

        if masking_type == 'crop': # [1, 512]
            x = vit(x)
            return x[:, 0, :]

        x = vit.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]

        # size = x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + vit.positional_embedding.to(x.dtype)
        # x = x + self.original_pos_embedding
        x = vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        L, N, D = x[1:, :, :].size(0), x.size(1), x.size(2)
        size = int(np.sqrt(L))
        assert size * size == L

        pred_masks = TF.resize(pred_masks.type(torch.float32), (size, size))

        if masking_type == 'token_masking':
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx >= masking_block:
                    cls = x[:1,:,:]
                    x = x[1:,:,:]
                    x = x.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]

                    x = x.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

                    x = torch.mul(x, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = x.size(0)
                    x = x.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]

                    x = x.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    x = torch.cat([cls.expand(-1,N,-1), x], dim=0) # [50, 46, 768]
                    x = resblock(x) # [50, 46, 768]

                    if block_idx == self.last_layer:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 512]

                else:
                    x = resblock(x)

        elif masking_type == 'specific_masking':
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx == masking_block:
                    cls = x[:1, :, :]
                    x = x[1:, :, :]
                    x = x.permute(1, 2, 0)  # [49, 1, 768] -> [1, 768, 49]

                    x = x.view(N, D, size, size).contiguous()  # [1, 768, 49] -> [1, 768, 7, 7]

                    x = torch.mul(x, pred_masks[:, None, :, :])  # [46, 768, 7, 7]
                    N = x.size(0)
                    x = x.view(N, D, L).contiguous()  # [46, 768, 7, 7] -> [46, 768, 49]

                    x = x.permute(2, 0, 1)  # [46, 768, 49] -> [49, 46, 768]
                    x = torch.cat([cls.expand(-1, N, -1), x], dim=0)  # [50, 46, 768]
                    x = resblock(x)  # [50, 46, 768]
                else:
                    x = resblock(x)

                if block_idx == self.last_layer:
                    x = x.permute(1, 0, 2) # [46, 50, 768]
                    x = self.model.visual.ln_post(x[:,0,:])
                    if self.model.visual.proj is not None:
                        x = x @ self.model.visual.proj




        elif masking_type == 'attn_masking':
            attn_mask = self.make_attn_mask(pred_masks)
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        N = pred_masks.shape[0]
                        x = x.expand(-1, N, -1)

                    x = resblock(x, attn_mask=attn_mask)

                    if block_idx == self.last_layer:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 512]

                else:
                    x = resblock(x)



        #
        # attn_mask = self.masks_to_attn_map(masks)
        # attn_mask = attn_mask.type(self.model.dtype)
        # num_masks = attn_mask.size(0)


        return x



class CLIPMaskedSpatialViT(nn.Module):
    def __init__(self, model_name='ViT-B/32', upsample=1, start_block=0, align_corners=None):
        super(CLIPMaskedSpatialViT, self).__init__()

        if model_name == 'ViT-B/32':
            self.target_size = 7
            self.patch_size = 32
        elif model_name == 'ViT-B/16':
            self.target_size = 14
            self.patch_size = 16
        elif model_name == 'ViT-L/14':
            self.target_size = 16
            self.patch_size = 14

        self.model, self.preprocess = clip.load(model_name)

        self.align_corners = align_corners

        assert (upsample == 1) or (upsample & (upsample-1) == 0)  # power of 2
        self.upsample = upsample
        self.target_size = self.target_size * self.upsample
        self.stem_stride = self.patch_size // upsample
        self.model.visual.conv1.stride = self.stem_stride
        self.model.visual.conv1.padding = (
            self.patch_size - 1) // 2  # TODO: make it more precise
        self.model.visual.positional_embedding = self.upsample_pos_emb(
            self.model.visual.positional_embedding)

        self.start_block = start_block

    @property
    def device(self):
        return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    def upsample_pos_emb(self, emb):
        # upsample the pretrained embedding for higher resolution
        # emb size NxD
        first = emb[:1, :]
        emb = emb[1:, :]
        N, D = emb.size(0), emb.size(1)
        size = int(np.sqrt(N))
        assert size * size == N
        new_size = size * self.upsample
        emb = emb.permute(1, 0)
        emb = emb.view(1, D, size, size).contiguous()
        emb = F.upsample(emb, size=new_size, mode='bilinear',
                         align_corners=self.align_corners)
        emb = emb.view(D, -1).contiguous()
        emb = emb.permute(1, 0)
        emb = torch.cat([first, emb], 0)
        emb = nn.parameter.Parameter(emb.half())
        return emb

    def masks_to_attn_map(self, masks):
        # masks size NxHxW
        N = masks.size(0)
        # masks is 1 for the object and 0 for others, need to invert it
        masks = 1 - masks.bool().float()
        # interpolate to target size
        masks = masks.unsqueeze(1).float()
        target_size = (self.target_size, self.target_size)
        masks = F.interpolate(masks, size=target_size,
                              mode='nearest', align_corners=None)
        masks = masks.squeeze(1)
        attn_map = masks.view(N, -1)
        attn_map = torch.cat([attn_map, 1-torch.eye(N).to(attn_map.device)], 1)
        attn_map = attn_map.bool().half() * (-100)
        return attn_map

    def encode_text(self, text):
        return self.model.encode_text(text)

    def encode_image(self, image):
        return self.model.encode_image(image)

    def forward(self, im, masks):
        vit = self.model.visual
        x = im.type(self.model.dtype)

        x = vit.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        attn_mask = self.masks_to_attn_map(masks)
        attn_mask = attn_mask.type(self.model.dtype)
        num_masks = attn_mask.size(0)
        for block_idx, resblock in enumerate(vit.transformer.resblocks):
            if block_idx == self.start_block:
                gv = x[:1]
                gv = gv.repeat(num_masks, 1, 1)  # LND
            if block_idx >= self.start_block:
                attn = resblock.attn
                source = resblock.ln_1(torch.cat([x[1:], gv], 0))
                gv = gv + attn(
                    source[-num_masks:],
                    source,
                    source,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
                gv = gv + resblock.mlp(resblock.ln_2(gv))
            x = resblock(x)

        gv = gv.permute(1, 0, 2)
        gv = vit.ln_post(gv)
        if vit.proj is not None:
            gv = (gv.view(-1, gv.size(-1)) @
                  vit.proj).view(gv.size(0), gv.size(1), -1)

        return gv # image_features

    def get_mask_feature(self, image, pred_masks):
        with torch.no_grad():
            image = image.to(self.device)
            image_features = self.forward(image, pred_masks) # [1,46,512]

            # image_features = image_features.permute(1,2,0)

            image_features = image_features / image_features.norm(dim=2, keepdim=True)

        return image_features # [46,512,1]


    def get_text_feature(self, text):
        text_features = self.model.encode_text(text).unsqueeze(-1) #[batch, channel, 1]

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    def calculate_similarity_score(self, image_features, text_features):
        # image_features = [N, 1,1024]
        # text_feature = [1,1024,1]
        # logit_scale.exp() = 100. ?? 이거는 학습에 사용되는 건데 inference시에도 곱해줄 필요가 있을까?

        # image_features = [N, 512, 1]
        # text_features = [1, 512, 1]
        # image_features = image_features.squeeze(2)

        text_features = text_features.squeeze(-1)



        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()

        # logits_per_image = logit_scale * image_features @ text_features.t() # 16.5868

        logits_per_image = logit_scale * image_features @ text_features.t() # 0.1659
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image.squeeze(0) # [1, N, 1]