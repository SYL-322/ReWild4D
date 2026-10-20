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

import os
import json
from scene.dataset_readers import readArtemisInfo, load_prior
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from plyfile import PlyData, PlyElement
import numpy as np
import torch

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None,
                 resolution_scales=[1.0], load_flip=False):
        """
        Loads the cameras of args.source_path and either initializes the Gaussians from the Fauna shape (training)
        or loads the trained Gaussians of load_iteration (-1: "best").
        load_flip: also load the flip camera series
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.data_path = args.source_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = 'best'
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        scene_info = readArtemisInfo(args.source_path, args.white_background, load_flip)

        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append({k: camera_to_JSON(id, cam[k]) for k in cam.keys()})
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)


        self.cameras_extent = 1.0
        self.v_init = scene_info.v_init
        self.skin_weights = scene_info.skin_weights

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.v_init))
        else:
            self.gaussians.create_from_obj(scene_info.v_init, init_scale=args.init_scale)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_deform_ply(self, gaussian_tuple, iteration, save_path=None, save_render_version=False):
        if save_path is None:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            save_path = os.path.join(point_cloud_path, "deformed_point_cloud.ply")
            os.makedirs(point_cloud_path, exist_ok=True)

        xyz, scaling, rotation, opacity, sh_features = gaussian_tuple
        
            
        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self.gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.gaussians.inverse_opacity_activation(opacity).detach().cpu().numpy()
        scale = self.gaussians.scaling_inverse_activation(scaling).detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.gaussians.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(save_path)
        
        if save_render_version:
            scale_render = torch.log(scaling).detach().cpu().numpy()
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale_render, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(os.path.join(os.path.dirname(save_path), "3dgs_" + os.path.basename(save_path)))


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


def prepare_models(dataset: ModelParams, iteration: int):
    """load trained Gaussians + deform model of `iteration` from dataset.model_path"""
    gaussians = GaussianModel(dataset.sh_degree)
    v_init = torch.tensor(load_prior(dataset.source_path)).float()
    skin_weights_path = os.path.join(dataset.source_path, "fauna/W0.npy")
    skin_weights = np.load(skin_weights_path)
    skin_weights = torch.tensor(skin_weights)
    deform_model = DeformModel(v_init, skin_weights, dataset.voxel_deformer_res, use_code=dataset.use_pose_code)
    deform_model.load_weights(dataset.model_path, iteration=iteration)
    deform_model.update_status(0)
    gaussians.load_ply(os.path.join(dataset.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(iteration),
                                                 "point_cloud.ply"),
                                    og_number_points=len(v_init))
    
    xyz = gaussians.get_xyz
    num_gaussians = xyz.shape[0]
    print("Number of Gaussians:", num_gaussians)
    with open(f"{dataset.model_path}/gaussian_number.txt", "w") as f:
        f.write(str(num_gaussians))
        
    return gaussians, deform_model
