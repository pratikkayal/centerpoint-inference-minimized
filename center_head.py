

import copy 
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F




import centerpoint.utils.box_torch_ops as box_torch_ops
from centerpoint.utils.sequential_utils import Sequential
from centerpoint.utils.centernet_loss import FocalLoss, SmoothRegLoss, RegLoss
from centerpoint.utils.center_utils import ddd_decode, _transpose_and_gather_feat

# from det3d.models.builder import build_loss
from centerpoint.utils.weight_init import constant_init, kaiming_init
# from det3d.torchie.trainer import load_checkpoint

from torch.nn.modules.batchnorm import _BatchNorm

try:
    from dcn import DeformConv, ModulatedDeformConvPack
except:
    print("Deformable Convolution not built!")


"""
To build DCN:
python setup.py build_ext --inplace
"""


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()

    def forward(self, x,):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x

def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), depth, dtype=dtype, device=tensor.device
    )
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss

def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period

def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0.0):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    #dir_cls_targets = ((rot_gt - dir_offset) > 0).long()
    dir_cls_targets = (limit_period(rot_gt - dir_offset, 0.5, np.pi*2) >
                       0).long()
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets

def get_direction_target_center(reg_targets, one_hot=False, dir_offset=0.0):
    rot_gt = reg_targets[..., -1]
    dir_cls_targets = ((rot_gt - dir_offset) > 0).long()
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=reg_targets.dtype)
    return dir_cls_targets


def smooth_l1_loss(pred, gt, sigma):
    def _smooth_l1_loss(pred, gt, sigma):
        sigma2 = sigma ** 2
        cond_point = 1 / sigma2
        x = pred - gt
        abs_x = torch.abs(x)

        in_mask = abs_x < cond_point
        out_mask = 1 - in_mask

        in_value = 0.5 * (sigma * x) ** 2
        out_value = abs_x - 0.5 / sigma2

        value = in_value * in_mask.type_as(in_value) + out_value * out_mask.type_as(
            out_value
        )
        return value

    value = _smooth_l1_loss(pred, gt, sigma)
    loss = value.mean(dim=1).sum()
    return loss


def smooth_l1_loss_detectron2(input, target, beta: float, reduction: str = "none"):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
     """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss



def create_loss(
    loc_loss_ftor,
    cls_loss_ftor,
    box_preds,
    cls_preds,
    cls_targets,
    cls_weights,
    reg_targets,
    reg_weights,
    num_class,
    encode_background_as_zeros=True,
    encode_rad_error_by_sin=True,
    bev_only=False,
    box_code_size=9,
):
    batch_size = int(box_preds.shape[0])

    if bev_only:
        box_preds = box_preds.view(batch_size, -1, box_code_size - 2)
    else:
        box_preds = box_preds.view(batch_size, -1, box_code_size)

    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)

    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot_f(cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]

    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)

    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights
    )  # [N, M]

    return loc_losses, cls_losses


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"

class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads 
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv-1):
                fc.add(nn.Conv2d(in_channels, head_conv,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))    

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = dict()        
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

class DCNSepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_cls,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(DCNSepHead, self).__init__(**kwargs)

        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4) 
        
        self.feature_adapt_reg = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)  

        # heatmap prediction head 
        self.cls_head = Sequential(
            nn.Conv2d(in_channels, head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls,
                kernel_size=3, stride=1, 
                padding=1, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(init_bias)

        # other regression target 
        self.task_head = SepHead(in_channels, heads, head_conv=head_conv, bn=bn, final_kernel=final_kernel)


    def forward(self, x):    
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret



class CenterHead(nn.Module):
    def __init__(
        self,
        in_channels=[128,],
        tasks=[],
        dataset='nuscenes',
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        dcn_head=False,
    ):
        super(CenterHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights 
        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset

        self.in_channels = in_channels
        self.num_classes = num_classes

        # self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.box_n_dim = 9 if 'vel' in common_heads else 7  
        self.use_direction_classifier = False 

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

        # a shared convolution 
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )

        logger.info("Finish CenterHead Initialization")

    def forward(self, x, *kwargs):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            # hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.dataset in ['waymo', 'nuscenes']:
                if 'vel' in preds_dict:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['vel'], preds_dict['rot']), dim=1)  
                else:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['rot']), dim=1)   
                    target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]] # remove vel target                       
            else:
                raise NotImplementedError()

            ret = {}
 
            # Regression loss for dimension, offset, height, rotation    
            raise        
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight*loc_loss

            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, 'loc_loss_elem': box_loss.detach().cpu(), 'num_positive': example['mask'][task_id].float().sum()})

            rets.append(ret)
        
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []
        # if('double_flip' in test_cfg):
        #     double_flip = test_cfg['double_flip']
        # else:
        double_flip = False
        # double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # transform the prediction map back to their original coordinate befor flipping
                    # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
                    # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is 
                    # X and Y flip pointcloud(x=-x, y=-y).
                    # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
                    # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
                    # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1]) 
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[:4*int(batch_size):4]

            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)

                # y = -y reg_y = 1-reg_y
                batch_reg[:, 1, ..., 1] = 1 - batch_reg[:, 1, ..., 1]
                batch_reg[:, 2, ..., 0] = 1 - batch_reg[:, 2, ..., 0]

                batch_reg[:, 3, ..., 0] = 1 - batch_reg[:, 3, ..., 0]
                batch_reg[:, 3, ..., 1] = 1 - batch_reg[:, 3, ..., 1]
                batch_reg = batch_reg.mean(dim=1)

                # first yflip 
                # y = -y theta = pi -theta
                # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
                # batch_rots[:, 1] the same
                batch_rotc[:, 1] *= -1

                # then xflip x = -x theta = 2pi - theta
                # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
                # batch_rots[:, 2] the same
                batch_rots[:, 2] *= -1

                # double flip 
                batch_rots[:, 3] *= -1
                batch_rotc[:, 3] *= -1

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm.device).float()
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm.device).float()

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']

                if double_flip:
                    # flip vy
                    batch_vel[:, 1, ..., 1] *= -1
                    # flip vx
                    batch_vel[:, 2, ..., 0] *= -1

                    batch_vel[:, 3] *= -1
                    
                    batch_vel = batch_vel.mean(dim=1)

                batch_vel = batch_vel.reshape(batch, H*W, 2)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else: 
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            # if test_cfg.get('per_class_nms', False):
            #     pass 
            # else:
            rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range)) 

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list 

    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask 

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, scores, 
                                thresh=test_cfg.nms.nms_iou_threshold,
                                pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                post_max_size=test_cfg.nms.nms_post_max_size)
            # print('Selecting nms boxes now. Selected {} boxes out of {} with threshold = {}'.format(len(selected),len(boxes_for_nms),str(test_cfg.nms.nms_iou_threshold)))
            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts 

