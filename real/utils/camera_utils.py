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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from scene.dataset_readers import CameraInfo

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    gt_image = resized_image_rgb[:3, ...]

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    else:
        loaded_mask = PILtoTorch(cam_info.mask, resolution)
        
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  aux=cam_info.aux)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        if type(c) is dict:
            cam_dict = {}
            for k in c.keys():
                cam_dict[k] = loadCam(args, id, c[k], resolution_scale)
            camera_list.append(cam_dict)
        elif type(c) == CameraInfo:
            camera_list.append(loadCam(args, id, c, resolution_scale))
        else:
            # log error
            print("Error: cameraList_from_camInfos: camera info is not of type CameraInfo")
            exit()

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    # Fauna cameras keep their actual extrinsics in aux; R/T are renderer placeholders.
    if camera.aux is not None and 'Rot' in camera.aux and 'Trans' in camera.aux:
        Rt[:3, :3] = camera.aux['Rot']
        Rt[:3, 3] = camera.aux['Trans']

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    fx = fov2focal(camera.FovX, camera.width)
    fy = fov2focal(camera.FovY, camera.height)
    if camera.aux is not None and all(key in camera.aux for key in ['Rot', 'Trans', 'mvp']):
        projection = np.asarray(camera.aux['mvp']) @ W2C
        fx = float(abs(projection[0, 0]) * camera.width / 2)
        fy = float(abs(projection[1, 1]) * camera.height / 2)
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fy,
        'fx': fx
    }
    return camera_entry

