# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from .faster_rcnn import FasterRCNN
from typing import Dict, Tuple
from mmdet.structures import SampleList
import copy
import clip
import torch
from torch import Tensor
# from torchvision.utils import save_image
from mmengine.structures import InstanceData
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from mmdet.structures.bbox import bbox2roi
from torchvision.utils import save_image
import cv2
import numpy as np
import random
import os

normal_descriptions = [
    'a photo of many {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of a {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a photo of a nice {}.',
    'a bright photo of a {}.',
    'a bright photo of the {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of a small {}.',
    'a photo of the large {}.',
    'a photo of a clean {}.',
    'a photo of the cool {}.',
    'a photo of a cool {}.',
    'a close-up photo of a {}.',
    'a good photo of the {}.',
    'a close-up photo of the {}.',
    'a good photo of a {}.',
]
noisy_descriptions = [
    'a bad photo of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a photo of a hard to see {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a dark photo of a {}.',
    'the plastic {}.',
    'a black and white photo of the {}.',
    'a black and white photo of a {}.',
    'a pixelated photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a low resolution photo of a {}.',
    'a rendition of the {}.',
    'a rendition of a {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'itap of a {}.',
    'itap of my {}.',
    'a jpeg corrupted photo of the {}.',
    'a photo of the weird {}.',
    'a sculpture of a {}.',
    'a sculpture of the {}.',
    'a rendering of a {}.',
    'a rendering of the {}.',
    'graffiti of a {}.',
    'graffiti of the {}.',
    'a tattoo of a {}.',
    'a tattoo of the {}.',
    'the embroidered {}.',
    'a embroidered {}.',
    'a drawing of a {}.',
    'a drawing of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a {} in a video game.',
    'the {} in a video game.',
    'a doodle of a {}.',
    'a doodle of the {}.',
    'the origami {}.',
    'a origami {}.',
    'a sketch of a {}.',
    'a sketch of the {}.',
    'the toy {}.',
    'a toy {}.',
    'a cartoon {}.',
    'the cartoon {}.',
    'art of the {}.',
    'art of a {}.',
    'a plushie {}.',
    'the plushie {}.',
]


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.1, scale_factor=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor

    def forward(self, x1, x2):
        # 展平特征 [B, C, H, W] -> [B, C*H*W]
        x_1_flat = x1.view(x1.size(0), -1)
        x_2_flat = x2.view(x2.size(0), -1)
        
        # 特征归一化
        x_1_flat = F.normalize(x_1_flat, p=2, dim=1)
        x_2_flat = F.normalize(x_2_flat, p=2, dim=1)
        
        # 计算相似性矩阵 [B, B]
        sim_matrix = torch.mm(x_1_flat, x_2_flat.t())
        
        # 除以温度系数
        sim_matrix /= self.temperature
        
        # 构造标签
        batch_size = x_1_flat.size(0)
        labels = torch.arange(batch_size).to(x_1_flat.device)
        
        # 计算 InfoNCE 损失
        loss = F.cross_entropy(sim_matrix, labels)
        return self.scale_factor * loss


class FeatureProjector(nn.Module):
    def __init__(self, in_dim=256, out_dim=512):
        super(FeatureProjector, self).__init__()
        
        # 定义 MLP 网络
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),  # 256 → 1024
            nn.ReLU(),               # 激活函数
            nn.Linear(1024, out_dim)  # 1024 → 512
        )

    def forward(self, x):
        """
        :param x: 输入 (batch_size, 256, 7, 7)
        :return: 映射后的特征 (batch_size, 512)
        """
        # 1. Global Average Pooling (GAP) 变成 (batch_size, 256)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # 变成 (batch_size, 256, 1, 1)
        x = x.view(x.shape[0], -1)  # 变成 (batch_size, 256)

        # 2. 通过 MLP 投影到 (batch_size, 512)
        x = self.mlp(x)

        return x

@MODELS.register_module()
class FasterRCNNCBoost(FasterRCNN):#随机选择bbox，进行224*224缩放
    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 clip_model: str = 'RN101',
                 dataset_name: str = 'cityscapes',
                 temperature: float = 0.1,
                 scale_factor: float = 0.1) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        # self.clip_model, _ = clip.load(clip_model, device="cuda" if torch.cuda.is_available() else "cpu")
        # self.clip_model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.clip_model, self.preprocess = clip.load(clip_model, self.device, jit=False)
        self.clip_model, _ = clip.load(clip_model, device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        self.dataset_name = dataset_name
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.resize_transform = transforms.Resize((224, 224))
                    
        if dataset_name == 'cityscapes':
            self.class_name = ['person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']
            self.feature_projector = FeatureProjector(in_dim=256, out_dim=512)
        elif dataset_name == 'dwd':
            self.class_name = ['bus', 'bike', 'car', 'motor', 'person', 'rider', 'truck']
            self.feature_projector = FeatureProjector(in_dim=2048, out_dim=512)
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")
        
        self.info_nce_loss = InfoNCELoss(temperature=temperature, scale_factor=scale_factor)
        
    def merge_instance_data_lists(self, list1, list2):
        """合并两个InstanceData列表中的每个元素。"""
        merged_list = []

        # 假设两个列表长度相等，对每对InstanceData进行合并
        for data1, data2 in zip(list1, list2):
            merged_data = InstanceData()

            # 遍历第一个InstanceData中的属性
            for key in data1.keys():
                # 如果同一属性在两个数据中都有，则拼接它们
                if key in data2:
                    merged_data[key] = torch.cat([data1[key], data2[key]], dim=0)
                else:
                    merged_data[key] = data1[key]

            # 遍历第二个InstanceData，添加剩余的属性
            for key in data2.keys():
                if key not in merged_data:
                    merged_data[key] = data2[key]

            merged_list.append(merged_data)

        return merged_list
    
    def inspect_rpn_results(self, rpn_results_list):
        """打印每个 InstanceData 中各个 key 的长度。"""
        for i, instance_data in enumerate(rpn_results_list):
            print(f"\nInstanceData {i}:")
            for key in instance_data.keys():
                value = instance_data[key]
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape {value.shape}")
                else:
                    print(f"  {key}: {len(value)}")
    
    def average_rpn_losses(self, rpn_losses_1, rpn_losses_2):
        averaged_losses = {}

        for key in rpn_losses_1.keys():
            assert len(rpn_losses_1[key]) == len(rpn_losses_2[key]), "Loss lists must have the same length."

            averaged_losses[key] = [
                (loss_1 + loss_2) / 2 for loss_1, loss_2 in zip(rpn_losses_1[key], rpn_losses_2[key])
            ]

        return averaged_losses
    
    # def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
    #     """Extract features.

    #     Args:
    #         batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

    #     Returns:
    #         tuple[Tensor]: Multi-level features that may have
    #         different resolutions.
    #     """
    #     x = self.backbone(batch_inputs,)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x
    
    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        batch_inputs_ori = multi_batch_inputs['original']
        batch_inputs_aug = multi_batch_inputs['augment']

        batch_data_samples = multi_batch_data_samples['augment']
        # save_image(batch_inputs_ori[0], '/home/xuxiaoran/ori.jpg')
        # print('batch_data_samplestype',type(batch_data_samples[0]))
        x_ori = self.extract_feat(batch_inputs_ori)
        x_aug = self.extract_feat(batch_inputs_aug)
        
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            aug_rpn_losses, aug_rpn_results_list = self.rpn_head.loss_and_predict(
                x_aug, rpn_data_samples, proposal_cfg=proposal_cfg)
            ori_rpn_losses, ori_rpn_results_list = self.rpn_head.loss_and_predict(
                x_ori, rpn_data_samples, proposal_cfg=proposal_cfg)
         
            rpn_results_list = self.merge_instance_data_lists(
                aug_rpn_results_list, ori_rpn_results_list)
            rpn_losses = self.average_rpn_losses(aug_rpn_losses, ori_rpn_losses)
        
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_losses = self.roi_head.loss(x_aug, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        # print('x_aug1',x_aug[0].shape)
        batch_data_samples_copy = copy.deepcopy(batch_data_samples)
        # print('batch_data_samples_copy',batch_data_samples_copy)
        bbox_aug_feats, bbox_ori_feats, bg_aug_feats, bg_ori_feats, selected_labels = self.random_bbox_feats(batch_inputs_ori, batch_inputs_aug, batch_data_samples_copy)
        
        loss_mi = self.calculate_loss(bbox_aug_feats, bbox_ori_feats, bg_aug_feats, bg_ori_feats, selected_labels)
        loss_mi = {'loss_mi': loss_mi}
        losses.update(loss_mi)

        return losses
    
    # def calculate(self, bbox_aug_feats, bbox_ori_feats, bg_feats):
       
    #     loss_mi = self.caulate_loss(bbox_aug_feats, bbox_ori_feats, bg_feats)

    #     return loss_mi
    
    def random_bbox_feats(self, batch_inputs_ori, batch_inputs_aug, batch_data_samples):
        #mining hard examples
        # background_images = []
        
        # selected_bbox_results = []
        # selected_bbox_results1 = []
        # bbox_feats_batch = []
        # bbox_feats1_batch = []
        cropped_aug_images = []
        cropped_ori_images = []
        bg_aug_images = []
        bg_ori_images = []
        selected_labels = []

        # bg_feats = []
        # bg_feats1 = []
        for i, sample in enumerate(batch_data_samples):
            gt_instances = sample.gt_instances
            labels = gt_instances.labels
            # print('labels',labels)
            unique_labels = torch.unique(labels)
            bboxes = gt_instances.bboxes
            selected_bboxes = []
            # selected_labels = []
            aug_image = batch_inputs_aug[i]
            ori_image = batch_inputs_ori[i]
            # print('bboxes',bboxes.shape)
            if labels.numel() == 0:
                # zero_row = torch.zeros(bboxes.shape[0], 1, device=bboxes.device)
                # rois = torch.cat([zero_row, bboxes], dim=1)
                # with torch.no_grad():
                #     bbox_feats = self.roi_head._bbox_forward(x, rois)
                #     bbox_feats1 = self.roi_head._bbox_forward(x1, rois)
                # selected_bbox_results.append(bbox_feats)
                # selected_bbox_results1.append(bbox_feats1)
                # print('11bbox_feats[0]',bbox_feats[0].shape)
                
                _, H, W = aug_image.shape
                start_x = random.randint(0, max(0, W - 100))
                start_y = random.randint(0, max(0, H - 100))
                start_x_bbox = random.randint(0, max(0, W - 100))
                start_y_bbox = random.randint(0, max(0, H - 100))
                # 随机截取一个不超过 100x100 的张量
                end_x = start_x + random.randint(1, min(100, W - start_x))
                end_y = start_y + random.randint(1, min(100, H - start_y))
                end_x_bbox = start_x_bbox + random.randint(1, min(100, W - start_x_bbox))
                end_y_bbox = start_y_bbox + random.randint(1, min(100, H - start_y_bbox))
                
                background_aug_image = aug_image[:, start_y:end_y, start_x:end_x]
                background_ori_image = ori_image[:, start_y:end_y, start_x:end_x]
                bbox_aug_image = aug_image[:, start_y_bbox:end_y_bbox, start_x_bbox:end_x_bbox]
                bbox_ori_image = ori_image[:, start_y_bbox:end_y_bbox, start_x_bbox:end_x_bbox]                
                
                background_aug_image = self.resize_transform(background_aug_image)
                background_ori_image = self.resize_transform(background_ori_image)
                bbox_aug_image = self.resize_transform(bbox_aug_image)
                bbox_ori_image = self.resize_transform(bbox_ori_image)
                
                hard_label = 'background'
                selected_labels.append(hard_label)
                cropped_aug_images.append(bbox_aug_image)
                cropped_ori_images.append(bbox_ori_image)
                bg_aug_images.append(background_aug_image)
                bg_ori_images.append(background_ori_image)
                # with torch.no_grad():
                #     bg_feat = self.extract_feat(background_aug_image)
                #     bg_feat1 = self.extract_feat(background_ori_image)
                #     bbox_feats = self.extract_feat(bbox_aug_image)
                #     bbox_feats1 = self.extract_feat(bbox_ori_image)
                # bg_feats.append(bg_feat[-1])
                # bg_feats1.append(bg_feat1[-1])
                # selected_bbox_results.append(bbox_feats[-1])
                # selected_bbox_results1.append(bbox_feats1[-1])
                # print('11bbox_feats',bbox_feats['bbox_feats'].shape)
                # background_images.append(background_image)
            else:
                # # print('bboxes',bboxes.shape)
                # zero_row = torch.zeros(bboxes.shape[0], 1, device=bboxes.device)
                # rois = torch.cat([zero_row, bboxes], dim=1)
                # with torch.no_grad():
                #     bbox_results = self.roi_head._bbox_forward(x, rois)
                #     bbox_results1 = self.roi_head._bbox_forward(x1, rois)
                # # print('bbox_results',bbox_results.shape)
                # # 为每个标签随机选择一个样本
              
                for label in unique_labels:
                    # 获取当前标签的所有样本的索引
                    label_indices = torch.nonzero(labels == label).squeeze(-1)
                
                    # 随机选择一个样本作为 difficult_sample
                    selected_index = random.choice(label_indices)  # 随机选择一个样本
                    selected_bboxes.append(bboxes[selected_index])
                    selected_labels.append(self.class_name[labels[selected_index]])
                    # for bbox in bboxes[selected_index]:
                    # print('bboxes[selected_index]',bboxes[selected_index])
                    x1, y1, x2, y2 = bboxes[selected_index].int()
                    cropped_aug_image = aug_image[:, y1:y2, x1:x2]
                    cropped_ori_image = ori_image[:, y1:y2, x1:x2]
                    # print('cropped_aug_image',cropped_aug_image.shape)
                    cropped_aug_image = self.resize_transform(cropped_aug_image)
                    cropped_ori_image = self.resize_transform(cropped_ori_image)
                    # with torch.no_grad():
                    #     feature = self.extract_feat(cropped_aug_image)
                    #     feature1 = self.extract_feat(cropped_ori_image)
                    #     selected_bbox_results.append(feature[-1])
                    #     selected_bbox_results1.append(feature1[-1])
                    cropped_aug_images.append(cropped_aug_image)
                    cropped_ori_images.append(cropped_ori_image)
        
                selected_bboxes = torch.stack(selected_bboxes)
               
                ori_image = batch_inputs_ori[i]
                aug_image = batch_inputs_aug[i]

                _, H, W = aug_image.shape                
                # 生成不与任何 bbox 重叠的随机区域
                def get_random_background_bbox(H, W, bboxes, max_size=100):
                    while True:
                        start_x = random.randint(0, W - 1)
                        start_y = random.randint(0, H - 1)
                        end_x = start_x + random.randint(1, min(max_size, W - start_x))
                        end_y = start_y + random.randint(1, min(max_size, H - start_y))
                        
                        # 检查是否与任何 bbox 重叠
                        overlap = False
                        for bbox in bboxes:
                            x1, y1, x2, y2 = map(int, bbox.tolist())
                            if not (end_x <= x1 or start_x >= x2 or end_y <= y1 or start_y >= y2):
                                overlap = True
                                break
                        if not overlap:
                            return start_x, start_y, end_x, end_y

                # print('difficult_sample_bboxes',difficult_sample_bboxes)
                if selected_bboxes.ndim == 1:
                    selected_bboxes = selected_bboxes.unsqueeze(0)
                
                bg_x1, bg_y1, bg_x2, bg_y2 = get_random_background_bbox(H, W, selected_bboxes)
                background_aug_image = aug_image[:, bg_y1:bg_y2, bg_x1:bg_x2]
                background_aug_image = self.resize_transform(background_aug_image)
                background_ori_image = ori_image[:, bg_y1:bg_y2, bg_x1:bg_x2]
                background_ori_image = self.resize_transform(background_ori_image)
                # bg_feat = self.extract_feat(background_image.unsqueeze(0))
                # bg_bbox = torch.tensor([bg_x1, bg_y1, bg_x2, bg_y2], device=zero_row.device).unsqueeze(0)
                # # print('bg_bbox',bg_bbox.shape)
                bg_aug_images.append(background_aug_image)
                bg_ori_images.append(background_ori_image)
                # # zero_row_bg = torch.zeros(bg_bbox.shape[0], 1, device=bg_bbox.device)
                # with torch.no_grad():
                #     feature_bg_aug = self.extract_feat(background_aug_image)
                #     feature_bg_ori = self.extract_feat(background_ori_image)
                #     # selected_bbox_results.append(feature[0])
                #     # selected_bbox_results1.append(feature1[0])
                # bg_feats.append(feature_bg_aug[0])
                # bg_feats1.append(feature_bg_ori[0])
                # print('bg_feat',bg_feat['bbox_feats'].shape)
                del ori_image, aug_image
            del gt_instances, labels, bboxes
            torch.cuda.empty_cache()
        cropped_aug_images = torch.stack(cropped_aug_images)
        cropped_ori_images = torch.stack(cropped_ori_images)
        bg_aug_images = torch.stack(bg_aug_images)
        bg_ori_images = torch.stack(bg_ori_images)
        with torch.no_grad():
            feature = self.extract_feat(cropped_aug_images)
            feature1 = self.extract_feat(cropped_ori_images)
            bg_feat = self.extract_feat(bg_aug_images)
            bg_feat1 = self.extract_feat(bg_ori_images)
        return feature[-1], feature1[-1], bg_feat[-1], bg_feat1[-1], selected_labels

                
                
                
    def img_encoder(self, ori_cropped_images, aug_cropped_images):
        
        # print('ori_cropped_images_stack',ori_cropped_images.shape)
        
        ori_clip_features = self.clip_model.encode_image(ori_cropped_images)
        aug_clip_features = self.clip_model.encode_image(aug_cropped_images)
        return ori_clip_features, aug_clip_features
    
    def text_encoder(self, foreground_labels, imagenet_templates):
        text_targets = []
        for i, label in enumerate(foreground_labels):
            target = self.compose_text_with_templates(label, imagenet_templates)
            # print('target',target)
            tokens = clip.tokenize(target).to(self.device)
            
            text_target = self.clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
            text_target /= text_target.norm(dim=-1, keepdim=True)
            text_target = text_target.squeeze(0).type(torch.float32)  # 移除 batch 维度
            text_targets.append(text_target)
        text_targets = torch.stack(text_targets, dim=0)
        # print('text_targets',text_targets.shape)
        return text_targets
    
    def compose_text_with_templates(self, text: str, templates) -> list:
        return [template.format(text) for template in templates]
    
    # def caulate_loss(self, x_ori_cropped, x_aug_cropped, text_features):
    #     # 1. 提取特征
    #     # print('text_features',text_features.shape)
    #     # print('x_ori_cropped',x_ori_cropped.shape)
    #     ori_features = self.feature_projector(x_ori_cropped)
    #     aug_features = self.feature_projector(x_aug_cropped)

    #     # 2. 计算 InfoNCE 损失
    #     loss1 = self.info_nce_loss(ori_features, text_features)
    #     loss2 = self.info_nce_loss(aug_features, text_features)
    #     loss3 = self.info_nce_loss(ori_features, aug_features)
        
    #     loss = (loss1 + loss2 + loss3) / 3
    #     # print('lossmimim',loss)
    #     return loss
    # def extract_feat(self, batch_inputs: Tensor, out_stage=-1) -> Tuple[Tensor]:
    #     """Extract features.

    #     Args:
    #         batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

    #     Returns:
    #         tuple[Tensor]: Multi-level features that may have
    #         different resolutions.
    #     """
    #     x = self.backbone(batch_inputs, out_stage)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x
       
                
    
    # def caulate_loss(self, x_ori_cropped, x_aug_cropped, text_features):
    def calculate_loss(self, bbox_aug_feats, bbox_ori_feats, bg_aug_feats, bg_ori_feats, selected_labels):
        # 1. 提取特征
        # print('bbox_ori_feats',bbox_ori_feats.shape)
        # print('bbox_aug_feats',bbox_aug_feats.shape)
        # print('bg_feats',bg_feats.shape)
        # print('bbox_aug_feats, bbox_ori_feats, bg_aug_feats, bg_ori_feats',bbox_aug_feats.shape, bbox_ori_feats.shape, bg_aug_feats.shape, bg_ori_feats.shape)
        text_bg_features_noisy = self.text_encoder(['background'], noisy_descriptions)
        text_bg_features_normal = self.text_encoder(['background'], normal_descriptions)
        if text_bg_features_noisy.size(0) != bg_aug_feats.size(0):
            repeat_factor = bg_aug_feats.size(0)
            text_bg_features_noisy = text_bg_features_noisy.repeat(repeat_factor, 1)
            # print('text_bg_features_noisy after repeat', text_bg_features_noisy.shape)
        if text_bg_features_normal.size(0) != bg_ori_feats.size(0):
            repeat_factor = bg_ori_feats.size(0)
            text_bg_features_normal = text_bg_features_normal.repeat(repeat_factor, 1)
        # print('select_labels',selected_labels)
        
        text_features_noisy = self.text_encoder(selected_labels, noisy_descriptions)
        text_features_normal = self.text_encoder(selected_labels, normal_descriptions)
        # text_bg_features_normal = self.text_encoder(['background'], normal_descriptions)
        # print('bbox_ori_feats.squeeze(0)',bbox_ori_feats.squeeze(0).shape)
        # print('text_features_noisy',text_features_noisy.shape)
        # print('text_bg_features_noisy',text_bg_features_noisy.shape)
        ori_features = self.feature_projector(bbox_ori_feats)
        aug_features = self.feature_projector(bbox_aug_feats)
        # print('bg_aug_feats',bg_aug_feats.shape)
        bg_aug_features = self.feature_projector(bg_aug_feats)
        bg_ori_features = self.feature_projector(bg_ori_feats)
        # print('bg_aug_feats1111',bg_aug_features.shape)
        
        combined_aug_features = torch.cat((aug_features, bg_aug_features), dim=0)
        combined_ori_features = torch.cat((ori_features, bg_ori_features), dim=0)
        # print('combined_aug_features',combined_aug_features.shape)
        # print('combined_ori_features',combined_ori_features.shape)
        loss1 = self.info_nce_loss(combined_aug_features, combined_ori_features)
        
        loss2 = self.info_nce_loss(bg_aug_features, text_bg_features_noisy)
        loss3 = self.info_nce_loss(aug_features, text_features_noisy)
        loss4 = self.info_nce_loss(bg_ori_features, text_bg_features_normal)
        loss5 = self.info_nce_loss(ori_features, text_features_normal)
        # print('loss3',loss3)
        
        return (loss1+(loss2+loss3+loss4+loss5)/4)/2
        # print('bg_aug_features',bg_aug_features.shape)
        # print('text_bg_features_noisy',text_bg_features_noisy.shape)
        # loss2 = self.info_nce_loss(bg_aug_features, text_bg_features_noisy)
        # print('bg_ori_features',bg_ori_features.shape)
        # print('text_bg_features_normal',text_bg_features_normal.shape)
        # loss3 = self.info_nce_loss(bg_ori_features, text_bg_features_normal)
        # return (loss1+loss2)/2
        # return (loss1+(loss2+loss3)/2)/2
        # loss = (loss1 + loss2 + loss3) / 3
        # # print('lossmimim',loss)
        # return loss
    