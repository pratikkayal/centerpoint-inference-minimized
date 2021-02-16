import sys
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS
import os
#Added
import glob

class PCDDataset(PointCloudDataset):
    NumPointFeatures = 4  # x, y, z, intensity
  
    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        sample=False,
        nsweeps=1,
        load_interval=1,
        **kwargs,
    ):
        self.load_interval = load_interval 
        self.sample = sample
        self.nsweeps = nsweeps
        print("Using {} sweeps".format(nsweeps))
        super(PCDDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self._info_path = info_path
        self._class_names = class_names
        self._num_point_features = PCDDataset.NumPointFeatures if nsweeps == 1 else PCDDataset.NumPointFeatures+1

    def reset(self):
        assert False 

    def load_infos(self, info_path):

        # with open(self._info_path, "rb") as f:
        #     _waymo_infos_all = pickle.load(f)
        #Added
        _waymo_infos_all = glob.glob(os.path.join(self._info_path,'*.pcd'))
        self._waymo_infos = _waymo_infos_all[::self.load_interval]

        print("Using {} Frames".format(len(self._waymo_infos)))

    def __len__(self):

        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self._info_path)

        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        pcd_name = self._waymo_infos[idx]

        info = {
            "path": pcd_name,
            "anno_path": self._info_path, 
            "token": pcd_name,
            "timestamp": None,
            "sweeps": []
        }

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps, 
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": pcd_name,
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "PCDDataset",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False):
        return NotImplementedError