#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
from .face_model import Resnet18
from .lipreading_models.lipreading_model import Lipreading
import os
import sys
# sys.path.insert(0, '..')
from .utils import load_json
from .wavenet import WaveNet


class ModelBuilder():
    # builder for facial attributes analysis stream
    def build_facial(self, pool_type='maxpool', fc_out=512, with_fc=False):
        pretrained = False
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet18(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)
        return net

    #builder for lipreading stream 
    def build_lipreadingnet(self):
        config_path = 'models/lipreading_models/lrw_snv1x_tcn2x.json'
        assert os.path.exists(config_path)
        args_loaded = load_json(config_path)
        print('Lipreading configuration file loaded.')
        tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                        'kernel_size': args_loaded['tcn_kernel_size'],
                        'dropout': args_loaded['tcn_dropout'],
                        'dwpw': args_loaded['tcn_dwpw'],
                        'width_mult': args_loaded['tcn_width_mult']}
        net = Lipreading(tcn_options=tcn_options,
                        backbone_type=args_loaded['backbone_type'],
                        relu_type=args_loaded['relu_type'],
                        width_mult=args_loaded['width_mult'],
                        extract_feats=True)

        return net

    def build_diffwave_model(self, model_cfg):
        cond_feat_size = 640        # size of feature dimension for the conditioner
        name = model_cfg.pop("_name_")
        model = WaveNet(cond_feat_size, **model_cfg)
        model_cfg["_name_"] = name # restore
        return model
