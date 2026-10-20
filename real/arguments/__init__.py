#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace, BooleanOptionalAction
import sys
import os
import yaml


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.voxel_deformer_res = 32
        self._source_path = ""
        self._model_path = ""
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.load2gpu_on_the_fly = False
        self.use_flip = True
        self.use_pose_code = True
        self.train_on_val_frames = False  # include is_val frames only when explicitly enabled
        self.deform_scale = True  # deform network also predicts per-Gaussian scale (False for APTv2)
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        # Fill settings absent from saved model arguments.
        for key, value in vars(self).items():
            key = key.lstrip("_")
            if getattr(g, key, None) is None:
                setattr(g, key, value)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.canonical_end = 500
        self.warm_up = 0  # start non-rigid refinement at this iteration
        self.iterations = 20000
        self.deform_lr_init = 0.001
        self.deform_lr_final = 0.0001
        self.position_lr_init = 0.005
        self.position_lr_final = 0.0005
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.deform_lr_max_steps = 20_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densify_interval = 100
        self.prune_interval = 100
        self.opacity_reset_interval = 1000
        self.densify_from_iter = 1000
        self.densify_until_iter = 18_000
        self.opacity_reset_until_iter = 20_000
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.05
        self.prune_max_screen_size = 0.0  # pixels; 0 disables screen-size pruning
        self.prune_min_scale_sum = 0  # prune if scale_x+scale_y+scale_z < this (0 = disabled)

        # Regularization weights
        self.use_image_reg = True
        self.lambda_image_reg = 0.2
        self.use_normal_reg = True
        self.lambda_normal_reg = 0.1
        self.optimize_voxel_weights = True
        self.voxel_weight_lr = 0.001
        self.use_skin_reg = True
        self.lambda_skin_reg = 0.1  # weight of (lambda_skin_tv * L_tv + lambda_skin_sparse * L_sparse)
        self.lambda_skin_tv = 100.0
        self.use_skin_sparse = True
        self.lambda_skin_sparse = 0.1
        self.lambda_compact = 0.1
        self.lambda_pose_sup = 1.0
        self.lambda_pose_trans = 0.1

        # Regularization schedules
        self.two_view_reg = False  # apply pose / stage-mask / normal / translation regs on both original and flip views
        self.pose_sup_until = 4000  # supervise predicted pose with the Fauna pose
        self.pose_trans_reg_from = 4000
        self.pose_trans_reg_until = 10000
        self.normal_reg_mode = "yaw_pitch"  # "yaw_pitch": random yaw + pitch; "spin": normal_reg_n_spin discrete yaw angles
        self.normal_reg_n_spin = 32
        self.normal_reg_from = 5000
        self.normal_reg_until = 10000
        self.skin_reg_until = 10000
        self.compact_until = 10000
        self.freeze_voxel_weights_at = -1  # stop optimizing voxel LBS weights at this iteration (-1: never)
        self.time_noise_steps = 20000  # decay steps of the time-input noise
        self.time_noise_std_frames = -1.0  # initial std in frame intervals; -1 uses 0.5 / training-camera count
        self.pose_noise_steps = 7000  # decay steps of the Fauna-pose mixing ratio

        super().__init__(parser, "Optimization Parameters")

    def extract(self, args):
        """Resolve the four cutoffs below: 0 ends with optimization; negatives count back from its end."""
        g = super().extract(args)
        for key in ["pose_trans_reg_until", "normal_reg_until", "skin_reg_until", "compact_until"]:
            configured_stop = getattr(g, key)
            if configured_stop <= 0:
                setattr(g, key, g.iterations + configured_stop)
        return g


def apply_config(parser: ArgumentParser, argv=None):
    """Read `--config xxx.yaml` and use its entries as parser defaults (command line still overrides)."""
    pre = ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args(argv)
    parser.add_argument("--config", type=str, default=None, help="yaml file with dataset-specific defaults")
    if known.config is None:
        return
    with open(known.config) as f:
        cfg = yaml.safe_load(f) or {}
    valid = {a.dest for a in parser._actions}
    unknown = set(cfg) - valid
    assert not unknown, f"unknown keys in {known.config}: {sorted(unknown)}"
    parser.set_defaults(**cfg)


def get_combined_args(parser: ArgumentParser):
    cmdline_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdline_string)

    cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
    if os.path.isfile(cfgfilepath):
        print("Config file found: {}".format(cfgfilepath))
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()
    else:
        print("Config file not found at {}, using command line arguments only".format(cfgfilepath))
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
