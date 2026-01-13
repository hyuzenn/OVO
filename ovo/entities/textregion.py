# Code based on TextRegion 
# https://github.com/avaxiao/TextRegion/blob/main/model_semantic.py

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

def resize_features(input_feature, crop_size, patch_size, points_per_h, points_per_w, crop_num_h, crop_num_w):
        
    bsz, _, embed_dim = input_feature.shape
    patch_num = crop_size // patch_size
    x_ori = input_feature.permute(0, 2, 1).contiguous().view(bsz, embed_dim, patch_num, patch_num)

    crop_id = 1
    x_multi_reso = F.interpolate(x_ori[:1], [points_per_h, points_per_w], mode="bilinear")
    for h_idx in range(crop_num_h):
        for w_idx in range(crop_num_w):
            y1 = h_idx * patch_num
            x1 = w_idx * patch_num
            y2 = y1 + patch_num
            x2 = x1 + patch_num

            x_multi_reso[:, :, y1:y2, x1:x2] = 0.5 * x_multi_reso[:, :, y1:y2, x1:x2] + x_ori[crop_id]
            crop_id += 1

    x_input = x_multi_reso.contiguous().view(1, embed_dim, crop_num_h * crop_num_w * patch_num ** 2).permute(0, 2, 1)
    return x_input


def remove_global_patch(x_input, feature_masks, global_patch_threshold):
    patch_norm = x_input.norm(dim=-1, keepdim=True)
    patch_features = (x_input / patch_norm)[0]
    patch_similarity = patch_features @ patch_features.T

    patch_2_region = patch_similarity @ (feature_masks > 0).float().T
    patch_2_region_avg = patch_2_region / (feature_masks > 0).sum(dim=-1)

    belong_score = patch_2_region_avg * (feature_masks > 0).float().T
    belong_score_avg = belong_score.sum(dim=-1) / ((feature_masks > 0).sum(dim=0) + 1e-9)

    outside_score = patch_2_region_avg * (feature_masks == 0).float().T
    outside_score_avg = outside_score.sum(dim=-1) / ((feature_masks == 0).sum(dim=0) + 1e-9)

    difference_score = (belong_score_avg - outside_score_avg).cpu().float().numpy()

    # Set the threshold for the difference score
    # threshold = difference_score[difference_score > 0].mean()
    feature_masks[:, difference_score < global_patch_threshold] = 0
    return feature_masks

class PETextRegion(torch.nn.Module):
    def __init__(self,
                 model,
                 model_card,
                 preprocess,
                 resize_method='multi_resolution',
                 remove_global_patch=True,
                 global_patch_threshold=0.07,
                 crop_size=None,
                 upsample_times=1,
                 mask_type='soft',
                 dtype='bf16',
                 device = torch.device("cuda") if torch.cuda.is_available() else "cpu",
                 project_and_normalize=True,
    ):
        super().__init__()

        if dtype == "fp32":
            dtype = torch.float32
        elif dtype == "bf16":
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.device = device
        self.dtype = dtype

        self.upsample_times = upsample_times
        self.mask_type = mask_type
        self.resize_method = resize_method
        self.remove_global_patch = remove_global_patch
        self.global_patch_threshold = global_patch_threshold
        self.project_and_normalize = project_and_normalize
        self.vlm = model
        if not model_card.startswith("PE"):
            raise NotImplementedError("Current TextRegion implementaion supports only PE models.")
        self.model_card = model_card
        self.clip_preprocess = preprocess
        try:
            self.patch_size = model.visual.patch_size
        except AttributeError:
            raise NotImplementedError("Only PE models from official repository are supported. OpenCLIP models are not supported!")

        if crop_size is None:
            try: 
                crop_size = int(self.model_card[-3:])
            except ValueError:
                crop_size = 224

        self.crop_size = crop_size

    def get_img_features(self, image):
        h, w = image.shape[1:3]

        if self.resize_method == 'multi_resolution':
            clip_inputs = []
            clip_inputs.append(
                self.clip_preprocess(
                    image)
                    )

            self.crop_num_h, self.crop_num_w = max(h // self.crop_size, 1), max(w // self.crop_size, 1)
            self.points_per_w = (self.crop_size // self.patch_size) * self.crop_num_w
            self.points_per_h = (self.crop_size // self.patch_size) * self.crop_num_h
            crop_size_h, crop_size_w = int(np.ceil(h / self.crop_num_h)), int(np.ceil(w / self.crop_num_w))
            assert self.crop_num_h * crop_size_h >= h and self.crop_num_w * crop_size_w >= w

            for h_idx in range(self.crop_num_h):
                for w_idx in range(self.crop_num_w):
                    y1 = h_idx * crop_size_h
                    x1 = w_idx * crop_size_w
                    y2 = min(y1 + crop_size_h, h)
                    x2 = min(x1 + crop_size_w, w)
                    y1 = max(y2 - crop_size_h, 0)
                    x1 = max(x2 - crop_size_w, 0)
                    crop_img = image[:, y1:y2, x1:x2]
                    clip_inputs.append(
                        self.clip_preprocess(
                        crop_img)
                        )

            clip_inputs = torch.stack(clip_inputs).to(self.device)

        else:
            self.points_per_w = self.crop_size // self.patch_size
            self.points_per_h = self.crop_size // self.patch_size
            clip_inputs = self.clip_preprocess(image).unsqueeze(0)
        
        pe_inputs = clip_inputs.to(device=self.device, dtype=self.vlm.visual.proj.dtype)
        last_blk_input = self.vlm.visual.forward_features(pe_inputs, norm=True)
        return last_blk_input
    
    def get_features_mask(self, region_masks):

        if self.upsample_times > 1:
            self.points_per_h *= self.upsample_times
            self.points_per_w *= self.upsample_times

        feature_masks = F.interpolate(region_masks.unsqueeze(0).float(), [self.points_per_h, self.points_per_w], mode="bilinear")
        feature_masks = feature_masks.reshape(-1, self.points_per_h * self.points_per_w)

        if self.mask_type == 'soft':
            feature_masks = torch.clamp(feature_masks, min=0, max=1)
        elif self.mask_type == 'hard':
            feature_masks = (feature_masks > 0).float()
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")
        
        return feature_masks

    def pe_value_with_sam2_attn(self, feature_masks, input_feature):

        if self.vlm.visual.use_cls_token:
            input_feature = input_feature[:, 1:]

        if self.resize_method == 'multi_resolution':
            x_input = resize_features(input_feature, self.crop_size, self.patch_size, self.points_per_h, self.points_per_w, self.crop_num_h, self.crop_num_w)
        else:
            x_input = input_feature

        if self.remove_global_patch:
            feature_masks = remove_global_patch(x_input, feature_masks, self.global_patch_threshold)

        batch = feature_masks.shape[0]

        assert x_input.shape[0] == 1 or x_input.shape[0] == feature_masks.shape[0]
        if x_input.shape[0] == 1:
            x = x_input.repeat(batch, 1, 1)
        else:
            x = x_input
        blk = self.vlm.visual.attn_pool
        q = blk.probe.repeat((batch, 1, 1)).to(x.dtype)
        k = blk.layernorm(x.mean(dim=-2, keepdim=True))
        k = k.repeat(1, x.shape[-2], 1).to(x.dtype)
        x = blk.attn(q, k, x, need_weights=False, key_padding_mask=feature_masks<=0)[0]

        if not self.project_and_normalize:
            return x.squeeze(1)
        
        with torch.no_grad():
            region_features =  x @ self.vlm.visual.proj
        region_features = F.normalize(region_features, dim=-1)
        return region_features.squeeze(1)

    def predict(self, image, region_masks):
        """
            - image (c, h, w) in [0,1]
        """
        last_blk_input = self.get_img_features(image)
        feature_masks = self.get_features_mask(region_masks)

        return self.pe_value_with_sam2_attn(feature_masks, last_blk_input)