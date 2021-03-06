


import argparse
import copy
import json
import logging
import os
# import pdb
import pickle 
import sys
import time
from collections import OrderedDict, defaultdict
from types import SimpleNamespace


try:
    import apex
except:
    print("No APEX!")

import torch.distributed as dist


import numpy as np
import six
import torch
import yaml
from argoverse.utils.json_utils import read_json_file, save_json_dict
from argoverse.utils.pkl_utils import load_pkl_dictionary, save_pkl_dictionary


# from det3d import __version__, torchie
# from det3d.datasets import build_dataloader, build_dataset

from centerpoint.utils.config import Config
from centerpoint.registry import DETECTORS

from centerpoint.dataset.centerpoint_dataloader import build_dataloader
from centerpoint.nuscenes_dataset import NuScenesDataset
from centerpoint.nuscenes_dataset import Reformat
from centerpoint.pcd import PCDDataset
from centerpoint.utils.checkpoint import load_checkpoint
from centerpoint.utils.compose import Compose
from centerpoint.utils.loading import LoadPointCloudAnnotations, LoadPointCloudFromFile
from centerpoint.utils.preprocess import AssignLabel, Preprocess
from centerpoint.utils.test_aug import DoubleFlip
from centerpoint.utils.voxel_generator import Voxelization

"""
python tools/dist_test.py configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset.py --work_dir work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset  --checkpoint work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset/epoch_20.pth  --speed_test 


other model
https://github.com/tianweiy/CenterPoint/blob/master/configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_flip.py
"""

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=log_level
        )
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    return logger


def example_to_device(example, device=None, non_blocking=False) -> dict:
    """ """
    assert device is not None

    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(v1).to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def parse_second_losses(losses):
    """ """
    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


def batch_processor(model, data, train_mode, **kwargs):
    """ """
    if "local_rank" in kwargs:
        device = torch.device(kwargs["local_rank"])
    else:
        device = None

    example = example_to_device(data, device, non_blocking=False)

    del data

    if train_mode:
        losses = model(example, return_loss=True)
        loss, log_vars = parse_second_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(example["anchors"][0])
        )
        return outputs
    else:
        return model(example, return_loss=False)




def build_detector(logger, cfg, train_cfg=None, test_cfg=None):

    registry = DETECTORS
    default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    return build_from_cfg(logger, cfg, registry, default_args)


def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)


# def build_from_cfg(logger, cfg, registry, default_args=None):
#     """Build a module from config dict.
#     Args:
#         cfg (dict): Config dict. It should at least contain the key "type".
#         registry (:obj:`Registry`): The registry to search the type from.
#         default_args (dict, optional): Default initialization arguments.
#     Returns:
#         obj: The constructed object.
#     """

#     from center_head import CenterHead
#     from centerpoint.models.scn_backbone import SpMiddleResNetFHD
#     from centerpoint.models.rpn import RPN
#     from centerpoint.models.voxel_encoder import VoxelFeatureExtractorV3
#     from centerpoint.models.voxelnet import VoxelNet

#     reader = VoxelFeatureExtractorV3(
#         num_input_features = 5,
#         norm_cfg = None
#     )
#     backbone = SpMiddleResNetFHD(
#         num_input_features = 5,
#         ds_factor = 8,
#         norm_cfg = None
#     )
#     neck = RPN(
#         layer_nums = [5, 5],
#         ds_layer_strides = [1, 2],
#         ds_num_filters = [128, 256],
#         us_layer_strides = [1, 2],
#         us_num_filters = [256, 256],
#         num_input_features = 256,
#         norm_cfg = None,
#         logger = logger # <Logger RPN (INFO)>
#     )
#     bbox_head = CenterHead(
#         mode = '3d',
#         in_channels = 512,
#         norm_cfg = None,
#         tasks = [
#             {'num_class': 1, 'class_names': ['car']},
#             {'num_class': 2, 'class_names': ['truck', 'construction_vehicle']},
#             {'num_class': 2, 'class_names': ['bus', 'trailer']},
#             {'num_class': 1, 'class_names': ['barrier']},
#             {'num_class': 2, 'class_names': ['motorcycle', 'bicycle']},
#             {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone']}
#         ],
#         dataset = 'nuscenes',
#         weight = 0.25,
#         code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
#         common_heads = {
#             'reg': (2, 2),
#             'height': (1, 2),
#             'dim': (3, 2),
#             'rot': (2, 2),
#             'vel': (2, 2)
#         },
#         encode_rad_error_by_sin = False,
#         direction_offset = 0.0,
#         share_conv_channel = 64,
#         dcn_head = True,
#         bn = True
#     )

#     detector = VoxelNet(
#         reader=reader,
#         backbone=backbone,
#         neck=neck,
#         bbox_head=bbox_head,
#         train_cfg=None,
#         test_cfg=SimpleNamespace(**default_args['test_cfg']),
#         pretrained=None
#     )
#     return detector

def build_from_cfg(logger, cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """

    from center_head import CenterHead
    from centerpoint.models.scn_backbone import SpMiddleResNetFHD
    from centerpoint.models.rpn import RPN
    from centerpoint.models.voxel_encoder import VoxelFeatureExtractorV3
    from centerpoint.models.voxelnet import VoxelNet

    reader = VoxelFeatureExtractorV3(
        num_input_features = 5,
        norm_cfg = None
    )
    backbone = SpMiddleResNetFHD(
        num_input_features = 5,
        ds_factor = 8,
        norm_cfg = None
    )
    neck = RPN(
        layer_nums = [5, 5],
        ds_layer_strides = [1, 2],
        ds_num_filters = [128, 256],
        us_layer_strides = [1, 2],
        us_num_filters = [256, 256],
        num_input_features = 256,
        norm_cfg = None,
        logger = logger # <Logger RPN (INFO)>
    )
    bbox_head = CenterHead(
        in_channels = sum([256, 256]),
        tasks=[
            dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']),
        ],
        dataset = 'waymo',
        weight = 0.25,
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads = {
            'reg': (2, 2),
            'height': (1, 2),
            'dim': (3, 2),
            'rot': (2, 2),
            'vel': (2, 2)
        },
        share_conv_channel = 64,
        dcn_head = False,
    )

    detector = VoxelNet(
        reader=reader,
        backbone=backbone,
        neck=neck,
        bbox_head=bbox_head,
        train_cfg=None,
        test_cfg=SimpleNamespace(**default_args['test_cfg']),
        pretrained=None
    )
    return detector


def build_dataset(cfg, args):
    """ """
    # pdb.set_trace()
    # print(cfg)
    nsweeps = cfg.nsweeps
    dataset_name = cfg.type
    split = args.split
    if split == 'test':
        info_path = f'data/{dataset_name}/infos_test_{str(nsweeps).zfill(2)}sweeps_withvelo.pkl'
    else:
        info_path = f'data/{dataset_name}/infos_val_{str(nsweeps).zfill(2)}sweeps_withvelo_filter_True.pkl'
    info_path = cfg.root_path

    # pipeline = [
    #         LoadPointCloudFromFile(dataset = 'NuScenesDataset')
    # ]
    pipeline = [
            LoadPointCloudFromFile(dataset = 'PCDDataset')
    ]
    if split != 'test':
        # only relevant for train or val
        pipeline += [LoadPointCloudAnnotations(with_bbox = True)]

    pipeline.extend([
            Preprocess(
                cfg=SimpleNamespace(**{
                    'mode': 'val',
                    'shuffle_points': False,
                    'remove_environment': False,
                    'remove_unknown_examples': False
                })
            ),
            # DoubleFlip(),
            Voxelization(
                cfg = cfg['pipeline'][3]['cfg']
                # SimpleNamespace(**{
                #     'range': [-54, -54, -5.0, 54, 54, 3.0],
                #     'voxel_size': [0.075, 0.075, 0.2],
                #     'max_points_in_voxel': 10,
                #     'max_voxel_num': 90000,
                #     'double_flip': True
                # })
            ),
            AssignLabel(
                cfg = cfg['pipeline'][4]['cfg']
                # SimpleNamespace(**{
                #     'target_assigner': SimpleNamespace(**{
                #         'tasks': cfg.tasks
                #         # [
                #         #     # per CBGS methodology
                #         #     SimpleNamespace(**{'num_class': 1, 'class_names': ['car']}),
                #         #     SimpleNamespace(**{'num_class': 2, 'class_names': ['truck', 'construction_vehicle']}),
                #         #     SimpleNamespace(**{'num_class': 2, 'class_names': ['bus', 'trailer']}),
                #         #     SimpleNamespace(**{'num_class': 1, 'class_names': ['barrier']}),
                #         #     SimpleNamespace(**{'num_class': 2, 'class_names': ['motorcycle', 'bicycle']}),
                #         #     SimpleNamespace(**{'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone']}),
                #         # ]
                #     }),
                #     'out_size_factor': 8,
                #     'dense_reg': 1,
                #     'gaussian_overlap': 0.1,
                #     'max_objs': 500,
                #     'min_radius': 2
                # })
            ),
            Reformat(double_flip=False) #Changed from True
        ])
    
    # dataset = NuScenesDataset(
    #     info_path = info_path,
    #     root_path = f'data/{dataset_name}/v1.0-test',
    #     test_mode = True,
    #     class_names = [
    #         'car',
    #         'truck',
    #         'construction_vehicle',
    #         'bus',
    #         'trailer',
    #         'barrier',
    #         'motorcycle',
    #         'bicycle',
    #         'pedestrian',
    #         'traffic_cone'
    #     ],
    #     nsweeps = nsweeps,
    #     ann_file = info_path,
    #     pipeline = pipeline
    # )
    dataset = PCDDataset(
        info_path = info_path,
        root_path = "data/Panasonic/processed/lidar",
        test_mode = True,
        class_names = cfg.class_names,
        # [
        #     'car',
        #     'truck',
        #     'construction_vehicle',
        #     'bus',
        #     'trailer',
        #     'barrier',
        #     'motorcycle',
        #     'bicycle',
        #     'pedestrian',
        #     'traffic_cone'
        # ],
        nsweeps = nsweeps,
        ann_file = info_path,
        pipeline = pipeline
    )

    # pdb.set_trace()
    # example = dataset[1]


    return dataset


def load_opts():
    """ """
    opts_dict = {
        'config': 'configs/mixed_pcd.py',#nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset.py',
        'work_dir': 'work_dirs/mixed_wn',
        'checkpoint': 'work_dirs/mixed_wn/latest.pth',
        'txt_result': False,
        'gpus': 1,
        'launcher': 'none',
        'speed_test': True,
        'local_rank': 0,
        'testset': False
    }
    opts = SimpleNamespace(**opts_dict)

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(opts.local_rank)

    return opts


def main(args):
    """ """
    opts = load_opts()

    # pdb.set_trace()
    cfg = Config.fromfile(opts.config)
    cfg.local_rank = opts.local_rank

    # update configs according to CLI args
    if opts.work_dir is not None:
        cfg.work_dir = opts.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    cfg.gpus = opts.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # pdb.set_trace()
    print('Model = ', cfg.model)
    model = build_detector(logger, cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if opts.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test, args)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val, args)
        
    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not opts.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # pdb.set_trace()
    checkpoint = load_checkpoint(model, opts.checkpoint, map_location="cpu")

    model = model.cuda()
    model.eval()
    mode = "val"

    logger.info(f"work dir: {opts.work_dir}")
    # if cfg.local_rank == 0:
    #     prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 

    for i, data_batch in enumerate(data_loader):
        """
        data_batch will have collated examples
        so for double flip, get an extra dimension on the outside, indexed with 0, 1, 2, 3 etc for 4 examples
        """
        print(f'{i}/{len(data_loader)}')
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=opts.local_rank,
            )
        for output in outputs:
            
            # `output` is a dictionary with keys
            # dict_keys(['box3d_lidar', 'scores', 'label_preds', 'metadata'])
            # box3d_lidar is a tensor of shape [217, 9]
            # scores is a tensor of shape [217]
            # label_preds is a tensor of shape [217]
            # metadata has {
            #     'image_prefix': PosixPath('data/nuScenes/v1.0-test'),
            #     'num_point_features': 5,
            #     'token': '3c61eda50f694585bac0614e4660eca1'
            #    }
            token = output["metadata"]["token"]
            detections[token] = data_batch["annos"]
            for k, v in output.items():
                if k not in ["metadata"]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
            detections[token]["annos"] = data_batch["annos"]
            # optionally dump the input!
            # detections[token]["input_coordinates"] = data_batch["coordinates"].to(cpu_device)
            # detections[token]["input_voxels"] = data_batch["voxels"].to(cpu_device)

    all_predictions = detections

    print("\n Total time per frame: ", (time_end -  time_start) / (end - start))

    if opts.local_rank != 0:
        return

    predictions = all_predictions

    if not os.path.exists(opts.work_dir):
        os.makedirs(opts.work_dir)
    
    pkl_fpath = os.path.join(opts.work_dir, args.pkl_save_fname)
    save_pkl_dictionary(pkl_fpath, predictions)
    predictions = load_pkl_dictionary(pkl_fpath)
    
    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=opts.work_dir, testset=opts.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if opts.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split',
        default='test',
        type=str,
        choices = ['val', 'test'],
        help="if val, will process detection labels"
    )
    parser.add_argument(
        '--dataset_name',
        default='argoverse',
        type=str,
        help="which dataset does this input data come from?, e.g. `argoverse`, `nuScenes`, etc"
    )
    parser.add_argument(
        '--pkl_save_fname',
        default='2020-01-13-argoverse_test_prediction.pkl',
        type=str,
        help="name of pickle file where predictions will be saved to"
    )
    
    args = parser.parse_args()
    
    if "argoverse" in args.dataset_name:
        args.nsweeps = 5 # 10 hz LiDAR, so 100 ms per sweep, cover 500 ms
    
    elif args.dataset_name == "nuScenes":
        args.nsweeps = 10 # 20 hz LiDAR, so 50 ms per sweep, cover 500 ms
    args.nsweeps = 2
    # args.pkl_save_fname = 'predictions.pkl'
    main(args)


