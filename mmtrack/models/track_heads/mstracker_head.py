# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.core import build_assigner, build_bbox_coder, build_sampler
from mmdet.core.anchor import build_prior_generator
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models import HEADS, build_loss

from mmtrack.core.track import depthwise_correlation

import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

@HEADS.register_module()
class MCorrelationHead(BaseModule):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 **kwargs):
        super(MCorrelationHead, self).__init__(init_cfg)
        self.kernel_convs = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.search_convs = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.head_convs = nn.Sequential(
            ConvModule(
                in_channels=mid_channels*5,
                out_channels=mid_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                act_cfg=None))
        
        self.RoI_align = torchvision.ops.RoIAlign(output_size=(7,7), spatial_scale=125/1024, sampling_ratio=-1)

    def cyclic_shift(self, fmap, dim=2, shift=0):
        index_array = torch.range(0, fmap.shape[dim]-1, 1).cuda().long()
        if shift < 0:
            shift += fmap.shape[dim]
        index_array = torch.cat((index_array[shift:], index_array[:shift]))
        # print(index_array)
        shift_map = torch.index_select(fmap, dim, index_array)
        return shift_map

    # def bbox_scale_down(self, bound, bbox_list, scale):
    #     bbox_list_scaledown = bbox_list * scale
    #     if bbox_list_scaledown[0] < 0:
    #         bbox_list_scaledown[0] = 0
    #     if bbox_list_scaledown[1] < 0:
    #         bbox_list_scaledown[1] = 0
    #     if bbox_list_scaledown[0] > bound[0]:
    #         bbox_list_scaledown[0] = bound[0]
    #     if bbox_list_scaledown[1] > bound[1]:
    #         bbox_list_scaledown[1] = bound[1]
    #     bbox_list_scaledown = torch.round(bbox_list_scaledown)
    #     return bbox_list_scaledown.int()


    def forward(self, kernel, search, bbox_list):
        # print(bbox_list)
        # bbox_list_scaledown = self.bbox_scale_down([kernel.shape[2], kernel.shape[3]], bbox_list, scale=125/1024)
        # print(bbox_list_scaledown)

        # print("kernel: ", kernel.shape)
        # print("search: ", search.shape)
        kernel = self.kernel_convs(kernel)
        search = self.search_convs(search)
        # print("kernel: ", kernel.shape)
        # print("search: ", search.shape)
        
        # # plotting  
        # kernel_vis = kernel.sum(0).sum(0).cpu().numpy()
        # for y in range(bbox_list_scaledown[0], bbox_list_scaledown[0]+bbox_list_scaledown[2]):
        #     for x in range(bbox_list_scaledown[1], bbox_list_scaledown[1]+bbox_list_scaledown[3]):
        #         kernel_vis[x][y] = 0
        # sns.heatmap(kernel_vis, vmin=0, vmax=150)
        # plt.savefig("/zhzhao/code/mmtracking_master_20220513/sys_log/kernel.png") 
        # plt.close()
        search_region = 2
        for shift_x in range(-1*search_region, search_region+1):
            for shift_y in range(-1*search_region, search_region+1):
                kernel_shift_x = self.cyclic_shift(kernel, dim=2, shift=shift_x)
                kernel_shift = self.cyclic_shift(kernel_shift_x, dim=3, shift=shift_y)
                # kernel_shift = torch.unsqueeze(kernel_shift, 0)
                # kernel_shift = torch.unsqueeze(kernel_shift, 0)
                if shift_y == -1*search_region and shift_y == -1*search_region:
                    kernel_shift_chunk = kernel_shift
                else:
                    kernel_shift_chunk = torch.cat((kernel_shift_chunk, kernel_shift), 1)

        # print("kernel_shift_chunk: ", kernel_shift_chunk.shape)
        # print("bbox_list:", bbox_list)

        bbox_list_xyxy = [torch.unsqueeze(b.clone(), 0).float() for b in bbox_list]
        for bi, bbox in enumerate(bbox_list_xyxy):
            bbox_list_xyxy[bi][0][0] = bbox[0][0] - bbox[0][2]/2
            bbox_list_xyxy[bi][0][1] = bbox[0][1] - bbox[0][3]/2
            bbox_list_xyxy[bi][0][2] = bbox[0][0] + bbox[0][2]/2
            bbox_list_xyxy[bi][0][3] = bbox[0][1] + bbox[0][3]/2
        # print(bbox_list_xyxy.size())
        # bbox_list_xyxy = torch.tensor(bbox_list_xyxy)
        output_roi = self.RoI_align(kernel_shift_chunk, bbox_list_xyxy)
        # # plotting  
        # sns.heatmap(kernel_shift.sum(0).sum(0).cpu().numpy(), vmin=0, vmax=150)
        # plt.savefig("/zhzhao/code/mmtracking_master_20220513/sys_log/kernel_shift.png") 
        # plt.close()

        # sns.heatmap(search.sum(0).sum(0).cpu().numpy(), vmin=0, vmax=150)
        # plt.savefig('/zhzhao/code/mmtracking_master_20220513/sys_log/search.png')
        # plt.close()

        # correlation_maps = depthwise_correlation(search, kernel)


        # print("output_roi: ", output_roi.shape)
        out = self.head_convs(output_roi)
        out = torch.unsqueeze(out, 1)
        # print("out: ", out.shape)
        return out


@HEADS.register_module()
class MSTrackerHead(BaseModule):
    """Siamese RPN head.

    This module is proposed in
    "SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks.
    `SiamRPN++ <https://arxiv.org/abs/1812.11703>`_.

    Args:
        anchor_generator (dict): Configuration to build anchor generator
            module.

        in_channels (int): Input channels.

        kernel_size (int): Kernel size of convs. Defaults to 3.

        norm_cfg (dict): Configuration of normlization method after each conv.
            Defaults to dict(type='BN').

        weighted_sum (bool): If True, use learnable weights to weightedly sum
            the output of multi heads in siamese rpn , otherwise, use
            averaging. Defaults to False.

        bbox_coder (dict): Configuration to build bbox coder. Defaults to
            dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1.]).

        loss_cls (dict): Configuration to build classification loss. Defaults
            to dict( type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)

        loss_bbox (dict): Configuration to build bbox regression loss. Defaults
            to dict( type='L1Loss', reduction='sum', loss_weight=1.2).

        train_cfg (Dict): Training setting. Defaults to None.

        test_cfg (Dict): Testing setting. Defaults to None.

        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 anchor_generator,
                 in_channels,
                 kernel_size=3,
                 norm_cfg=dict(type='BN'),
                 weighted_sum=False,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0., 0., 0., 0.],
                     target_stds=[1., 1., 1., 1.]),
                 loss_cls=dict(
                     type='CrossEntropyLoss', reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='L1Loss', reduction='sum', loss_weight=1.2),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(MSTrackerHead, self).__init__(init_cfg)
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner = build_assigner(self.train_cfg.assigner)
        self.sampler = build_sampler(self.train_cfg.sampler)
        self.fp16_enabled = False

        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for i in range(len(in_channels)):
            self.cls_heads.append(
                MCorrelationHead(in_channels[i], in_channels[i],
                                2 * self.anchor_generator.num_base_anchors[0],
                                kernel_size, norm_cfg))
            self.reg_heads.append(
                MCorrelationHead(in_channels[i], in_channels[i],
                                4 * self.anchor_generator.num_base_anchors[0],
                                kernel_size, norm_cfg))

        self.weighted_sum = weighted_sum
        if self.weighted_sum:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.reg_weight = nn.Parameter(torch.ones(len(in_channels)))


        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

    @auto_fp16()
    def forward(self, z_feats, x_feats, bbox_list):
        """Forward with features `z_feats` of exemplar images and features
        `x_feats` of search images.

        Args:
            z_feats (tuple[Tensor]): Tuple of Tensor with shape (N, C, H, W)
                denoting the multi level feature maps of exemplar images.
                Typically H and W equal to 7.
            x_feats (tuple[Tensor]): Tuple of Tensor with shape (N, C, H, W)
                denoting the multi level feature maps of search images.
                Typically H and W equal to 31.

        Returns:
            tuple(cls_score, bbox_pred): cls_score is a Tensor with shape
            (N, 2 * num_base_anchors, H, W), bbox_pred is a Tensor with shape
            (N, 4 * num_base_anchors, H, W), Typically H and W equal to 25.
        """
        assert isinstance(z_feats, tuple) and isinstance(x_feats, tuple)
        assert len(z_feats) == len(x_feats) and len(z_feats) == len(
            self.cls_heads)

        if self.weighted_sum:
            cls_weight = nn.functional.softmax(self.cls_weight, dim=0)
            reg_weight = nn.functional.softmax(self.reg_weight, dim=0)
        else:
            reg_weight = cls_weight = [
                1.0 / len(z_feats) for i in range(len(z_feats))
            ]

        cls_score = 0
        bbox_pred = 0
        for i in range(len(z_feats)):
            cls_score_single = self.cls_heads[i](z_feats[i], x_feats[i], bbox_list)
            bbox_pred_single = self.reg_heads[i](z_feats[i], x_feats[i], bbox_list)
            cls_score += cls_weight[i] * cls_score_single
            bbox_pred += reg_weight[i] * bbox_pred_single

        # print("cls_score: ", cls_score)
        # print("bbox_pred: ", bbox_pred)
        # print("cls_score: ", cls_score.size())
        # print("bbox_pred: ", bbox_pred.size())


        return cls_score, bbox_pred

    def _get_init_targets(self, gt_bbox, score_maps_size):
        """Initialize the training targets based on flattened anchors of the
        last score map."""
        num_base_anchors = self.anchor_generator.num_base_anchors[0]
        H, W = score_maps_size
        num_anchors = H * W * num_base_anchors
        labels = gt_bbox.new_zeros((num_anchors, ), dtype=torch.long)
        labels_weights = gt_bbox.new_zeros((num_anchors, ))
        bbox_weights = gt_bbox.new_zeros((num_anchors, 4))
        bbox_targets = gt_bbox.new_zeros((num_anchors, 4))
        return labels, labels_weights, bbox_targets, bbox_weights

    def _get_positive_pair_targets(self, gt_bbox, score_maps_size):
        """Generate the training targets for positive exemplar image and search
        image pair.

        Args:
            gt_bbox (Tensor): Ground truth bboxes of an search image with
                shape (1, 5) in [0.0, tl_x, tl_y, br_x, br_y] format.
            score_maps_size (torch.size): denoting the output size
                (height, width) of the network.

        Returns:
            tuple(labels, labels_weights, bbox_targets, bbox_weights): the
            shape is (H * W * num_base_anchors,), (H * W * num_base_anchors,),
            (H * W * num_base_anchors, 4), (H * W * num_base_anchors, 4)
            respectively. All of them are Tensor.
        """
        (labels, labels_weights, _,
         bbox_weights) = self._get_init_targets(gt_bbox, score_maps_size)

        if not hasattr(self, 'anchors'):
            self.anchors = self.anchor_generator.grid_priors(
                [score_maps_size], device=gt_bbox.device)[0]
            # Transform the coordinate origin from the top left corner to the
            # center in the scaled score map.
            feat_h, feat_w = score_maps_size
            stride_w, stride_h = self.anchor_generator.strides[0]
            self.anchors[:, 0:4:2] -= (feat_w // 2) * stride_w
            self.anchors[:, 1:4:2] -= (feat_h // 2) * stride_h

        anchors = self.anchors.clone()

        # The scaled feature map and the searched image have the same center.
        # Transform coordinate origin from the center to the top left corner in
        # the searched image.
        anchors += self.train_cfg.search_size // 2

        assign_result = self.assigner.assign(anchors, gt_bbox[:, 1:])
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bbox[:, 1:])
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        neg_upper_bound = int(self.sampler.num *
                              (1 - self.sampler.pos_fraction))
        if len(neg_inds) > neg_upper_bound:
            neg_inds = neg_inds[:neg_upper_bound]

        if len(pos_inds) > 0:
            labels[pos_inds] = 1
            labels_weights[pos_inds] = 1.0 / len(pos_inds) / 2
            bbox_weights[pos_inds] = 1.0 / len(pos_inds)

        if len(neg_inds) > 0:
            labels[neg_inds] = 0
            labels_weights[neg_inds] = 1.0 / len(neg_inds) / 2

        bbox_targets = self.bbox_coder.encode(
            anchors, gt_bbox[:, 1:].repeat(anchors.shape[0], 1))
        return labels, labels_weights, bbox_targets, bbox_weights

    def _get_negative_pair_targets(self, gt_bbox, score_maps_size):
        """Generate the training targets for negative exemplar image and search
        image pair.

        Args:
            gt_bbox (Tensor): Ground truth bboxes of an search image with
                shape (1, 5) in [0.0, tl_x, tl_y, br_x, br_y] format.
            score_maps_size (torch.size): denoting the output size
                (height, width) of the network.

        Returns:
            tuple(labels, labels_weights, bbox_targets, bbox_weights): the
            shape is (H * W * num_base_anchors,), (H * W * num_base_anchors,),
            (H * W * num_base_anchors, 4), (H * W * num_base_anchors, 4)
            respectively. All of them are Tensor.
        """
        (labels, labels_weights, bbox_targets,
         bbox_weights) = self._get_init_targets(gt_bbox, score_maps_size)
        H, W = score_maps_size
        target_cx, target_cy, _, _ = bbox_xyxy_to_cxcywh(gt_bbox[:, 1:])[0]
        anchor_stride = self.anchor_generator.strides[0]

        cx = W // 2
        cy = H // 2
        cx += int(
            torch.ceil((target_cx - self.train_cfg.search_size // 2) /
                       anchor_stride[0] + 0.5))
        cy += int(
            torch.ceil((target_cy - self.train_cfg.search_size // 2) /
                       anchor_stride[1] + 0.5))

        left = max(0, cx - 3)
        right = min(W, cx + 4)
        top = max(0, cy - 3)
        down = min(H, cy + 4)

        labels = labels.view(H, W, -1)
        labels[...] = -1
        labels[top:down, left:right, :] = 0

        labels = labels.view(-1)
        neg_inds = torch.nonzero(labels == 0, as_tuple=False)[:, 0]
        index = torch.randperm(
            neg_inds.numel(), device=neg_inds.device)[:self.train_cfg.num_neg]
        neg_inds = neg_inds[index]

        labels[...] = -1
        if len(neg_inds) > 0:
            labels[neg_inds] = 0
            labels_weights[neg_inds] = 1.0 / len(neg_inds) / 2
        labels[...] = 0

        return labels, labels_weights, bbox_targets, bbox_weights

    def get_targets(self, gt_bboxes, score_maps_size, is_positive_pairs):
        """Generate the training targets for exemplar image and search image
        pairs.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each
                search image with shape (1, 5) in [0.0, tl_x, tl_y, br_x, br_y]
                format.
            score_maps_size (torch.size): denoting the output size
                (height, width) of the network.
            is_positive_pairs (bool): list of bool denoting whether each ground
                truth bbox in `gt_bboxes` is positive.

        Returns:
            tuple(all_labels, all_labels_weights, all_bbox_targets,
            all_bbox_weights): the shape is (N, H * W * num_base_anchors),
            (N, H * W * num_base_anchors), (N, H * W * num_base_anchors, 4),
            (N, H * W * num_base_anchors, 4), respectively. All of them are
            Tensor.
        """
        (all_labels, all_labels_weights, all_bbox_targets,
         all_bbox_weights) = [], [], [], []

        for gt_bbox, is_positive_pair in zip(gt_bboxes, is_positive_pairs):
            if is_positive_pair:
                (labels, labels_weights, bbox_targets,
                 bbox_weights) = self._get_positive_pair_targets(
                     gt_bbox, score_maps_size)
            else:
                (labels, labels_weights, bbox_targets,
                 bbox_weights) = self._get_negative_pair_targets(
                     gt_bbox, score_maps_size)

            all_labels.append(labels)
            all_labels_weights.append(labels_weights)
            all_bbox_targets.append(bbox_targets)
            all_bbox_weights.append(bbox_weights)

        all_labels = torch.stack(all_labels)
        all_labels_weights = torch.stack(all_labels_weights) / len(
            all_labels_weights)
        all_bbox_targets = torch.stack(all_bbox_targets)
        all_bbox_weights = torch.stack(all_bbox_weights) / len(
            all_bbox_weights)

        return (all_labels, all_labels_weights, all_bbox_targets,
                all_bbox_weights)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, cls_score, bbox_pred, labels, labels_weights, bbox_targets,
             bbox_weights):
        """Compute loss.

        Args:
            cls_score (Tensor): of shape (N, 2 * num_base_anchors, H, W).
            bbox_pred (Tensor): of shape (N, 4 * num_base_anchors, H, W).
            labels (Tensor): of shape (N, H * W * num_base_anchors).
            labels_weights (Tensor): of shape (N, H * W * num_base_anchors).
            bbox_targets (Tensor): of shape (N, H * W * num_base_anchors, 4).
            bbox_weights (Tensor): of shape (N, H * W * num_base_anchors, 4).

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = {}
        N, _, H, W = cls_score.shape

        cls_score = cls_score.view(N, 2, -1, H, W)
        cls_score = cls_score.permute(0, 3, 4, 2, 1).contiguous().view(-1, 2)
        labels = labels.view(-1)
        labels_weights = labels_weights.view(-1)
        losses['loss_rpn_cls'] = self.loss_cls(
            cls_score, labels, weight=labels_weights)

        bbox_pred = bbox_pred.view(N, 4, -1, H, W)
        bbox_pred = bbox_pred.permute(0, 3, 4, 2, 1).contiguous().view(-1, 4)
        bbox_targets = bbox_targets.view(-1, 4)
        bbox_weights = bbox_weights.view(-1, 4)
        losses['loss_rpn_bbox'] = self.loss_bbox(
            bbox_pred, bbox_targets, weight=bbox_weights)

        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bbox_list(self, cls_score_list, bbox_pred_list, bbox_list, scale_factor):
        best_score_list = list()
        final_bbox_list = list()
        for i in range(len(cls_score_list)):
            best_score, final_bbox = self.get_bbox(cls_score_list[i], bbox_pred_list[i], bbox_list[i], scale_factor)
        best_score_list.append(best_score)
        final_bbox_list.append(final_bbox)
        return best_score_list, final_bbox_list

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bbox(self, cls_score, bbox_pred, prev_bbox, scale_factor):
        """Track `prev_bbox` to current frame based on the output of network.

        Args:
            cls_score (Tensor): of shape (1, 2 * num_base_anchors, H, W).
            bbox_pred (Tensor): of shape (1, 4 * num_base_anchors, H, W).
            prev_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] format.
            scale_factor (Tensr): scale factor.

        Returns:
            tuple(best_score, best_bbox): best_score is a Tensor denoting the
            score of `best_bbox`, best_bbox is a Tensor of shape (4, )
            with [cx, cy, w, h] format, which denotes the best tracked
            bbox in current frame.
        """
        score_maps_size = [(cls_score.shape[2:])]
        # print(cls_score.shape)
        if not hasattr(self, 'anchors'):
            self.anchors = self.anchor_generator.grid_priors(
                score_maps_size, device=cls_score.device)[0]
            # Transform the coordinate origin from the top left corner to the
            # center in the scaled feature map.
            feat_h, feat_w = score_maps_size[0]
            stride_w, stride_h = self.anchor_generator.strides[0]
            self.anchors[:, 0:4:2] -= (feat_w // 2) * stride_w
            self.anchors[:, 1:4:2] -= (feat_h // 2) * stride_h

        if not hasattr(self, 'windows'):
            self.windows = self.anchor_generator.gen_2d_hanning_windows(
                score_maps_size, cls_score.device)[0]

        H, W = score_maps_size[0]
        cls_score = cls_score.view(2, -1, H, W)
        cls_score = cls_score.permute(2, 3, 1, 0).contiguous().view(-1, 2)
        cls_score = cls_score.softmax(dim=1)[:, 1]

        bbox_pred = bbox_pred.view(4, -1, H, W)
        bbox_pred = bbox_pred.permute(2, 3, 1, 0).contiguous().view(-1, 4)
        bbox_pred = self.bbox_coder.decode(self.anchors, bbox_pred)
        bbox_pred = bbox_xyxy_to_cxcywh(bbox_pred)

        def change_ratio(ratio):
            return torch.max(ratio, 1. / ratio)

        def enlarge_size(w, h):
            pad = (w + h) * 0.5
            return torch.sqrt((w + pad) * (h + pad))

        # scale penalty
        scale_penalty = change_ratio(
            enlarge_size(bbox_pred[:, 2], bbox_pred[:, 3]) / enlarge_size(
                prev_bbox[2] * scale_factor, prev_bbox[3] * scale_factor))

        # aspect ratio penalty
        aspect_ratio_penalty = change_ratio(
            (prev_bbox[2] / prev_bbox[3]) /
            (bbox_pred[:, 2] / bbox_pred[:, 3]))

        # penalize cls_score
        penalty = torch.exp(-(aspect_ratio_penalty * scale_penalty - 1) *
                            self.test_cfg.penalty_k)
        penalty_score = penalty * cls_score

        # window penalty
        penalty_score = penalty_score * (1 - self.test_cfg.window_influence) \
            + self.windows * self.test_cfg.window_influence

        best_idx = torch.argmax(penalty_score)
        best_score = cls_score[best_idx]
        best_bbox = bbox_pred[best_idx, :] / scale_factor

        final_bbox = torch.zeros_like(best_bbox)

        # map the bbox center from the searched image to the original image.
        final_bbox[0] = best_bbox[0] + prev_bbox[0]
        final_bbox[1] = best_bbox[1] + prev_bbox[1]

        # smooth bbox
        lr = penalty[best_idx] * cls_score[best_idx] * self.test_cfg.lr
        final_bbox[2] = prev_bbox[2] * (1 - lr) + best_bbox[2] * lr
        final_bbox[3] = prev_bbox[3] * (1 - lr) + best_bbox[3] * lr

        return best_score, final_bbox
