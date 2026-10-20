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
import sys
import time
import random
from argparse import ArgumentParser, Namespace
from collections import defaultdict

import numpy as np
import torch
import json
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, apply_config
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import set_random_seed
from utils.logger_utils import AverageMeter, get_time_prefix, visual_schedule
from render_video import render_set_spin, render_set_spin_yaw_pitch
from utils.loss_utils import l1_loss, ssim, weighted_directional_tv_loss, bce_dice_loss
from utils.visual_utils import (render, rotate_gaussian, rotate_gaussian_yaw_pitch, skin_weight_colors,
                                vis_render, vis_spin_view, vis_canonical_view)


def loss_func(pred, gt, opt):
    Ll1 = l1_loss(pred, gt)
    Lssim = 1.0 - ssim(pred, gt)
    return (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim


def camera_schedule(total_frame, opt):
    """frame index for every iteration: 5 fixed frames during the canonical phase, then shuffled epochs"""
    if total_frame <= 0:
        raise ValueError('Training requires at least one camera pair')
    count = min(5, total_frame)
    start_frame = min(5, total_frame - count)
    cam_idx_list = list(range(start_frame, start_frame + count)) * int(np.ceil(opt.canonical_end / count))
    cam_idx_list = cam_idx_list[:opt.canonical_end]
    random.shuffle(cam_idx_list)
    num_epoch = int(np.ceil((opt.iterations - opt.canonical_end) / total_frame))
    for _ in range(num_epoch):
        cam_idx_list.extend(sorted(list(range(total_frame)), key=lambda k: random.random()))
    assert len(cam_idx_list) >= opt.iterations
    return cam_idx_list[:opt.iterations]


def masked_gt(view, bg):
    """input image with background replaced by bg, and its mask"""
    gt_image = view.original_image.cuda()
    gt_mask = view.original_mask.cuda()
    gt_image = torch.where(gt_mask == 0, bg[:, None, None], gt_image)  # (no boolean indexing: avoids a host sync)
    return gt_image, gt_mask


# Regularization
# reg_views: list of (camera, deformation stages, gt mask, Fauna pose) for the original view and,
# with opt.two_view_reg, the flipped view. Each loss is averaged over reg_views.

def pose_supervision_loss(deform_model, time_input, reg_views):
    """predicted bone transforms should match the Fauna pose"""
    loss = 0.0
    for view, _, _, view_pose in reg_views:
        pred_qua, pred_trans, _ = deform_model.deform.estimate_pose_forward(time_input, signal=view.aux['signal'])
        loss += torch.mean(torch.abs(view_pose - deform_model.deform.compose_transform(pred_qua, pred_trans)))
    return loss / len(reg_views)


def normal_smoothness_loss(opt, reg_views, gaussians, pipe, bg):
    """smooth rendered normals of the posed Gaussians seen from a random rotated viewpoint"""
    if opt.normal_reg_mode == "yaw_pitch":
        # Yaw in [0, 2*pi), pitch in [-0.25, 0.75] radians.
        yaw = np.random.uniform(0, 2 * np.pi)
        pitch = 0.25 + 0.5 * np.sin(np.random.uniform(0, 2 * np.pi))
        rotate_fn = lambda g: rotate_gaussian_yaw_pitch(g, yaw, pitch)
    elif opt.normal_reg_mode == "spin":
        # one of normal_reg_n_spin discrete yaw angles
        spin_angle = 2 * np.pi * np.random.randint(0, opt.normal_reg_n_spin) / opt.normal_reg_n_spin
        rotate_fn = lambda g: rotate_gaussian(g, spin_angle)
    else:
        raise ValueError(f"unknown normal_reg_mode {opt.normal_reg_mode}")
    loss = 0.0
    for view, view_stages, _, _ in reg_views:
        render_pkg = render(view, rotate_fn(view_stages[-1]), gaussians.active_sh_degree, gaussians.max_sh_degree, pipe, bg)
        loss += weighted_directional_tv_loss(render_pkg["normal"].unsqueeze(0), render_pkg["alpha"].unsqueeze(0), eps=1e-6)
    return loss / len(reg_views)


def skinning_loss(opt, deform_model, losses):
    """smooth (TV) and optionally sparse (low-entropy) voxel skinning weights"""
    L_skin = 0.0
    if opt.use_skin_sparse:
        L_sparse = deform_model.articulator.get_sparsity_loss()
        losses['skin_sparse'].update(L_sparse.item())
        L_skin += L_sparse * opt.lambda_skin_sparse
    L_tv = deform_model.articulator.get_tv_loss()
    losses['skin_tv'].update(L_tv.item())
    L_skin += L_tv * opt.lambda_skin_tv
    return L_skin


def compact_loss(deform_model, gaussians_stages, gaussians):
    """Gaussians bound to a bone should form a thin, limb-like cluster (posed and canonical)"""
    return deform_model.articulator.compute_compact_loss(gaussians_stages[0].xyz) \
        + deform_model.articulator.compute_compact_loss(gaussians.get_xyz)


def root_translation_loss(deform_model, time_input, reg_views):
    """keep the predicted bone translations small"""
    loss = 0.0
    for view, _, _, _ in reg_views:
        _, pred_trans, _ = deform_model.deform.estimate_pose_forward(time_input, signal=view.aux['signal'])
        loss += torch.mean(pred_trans ** 2)
    return loss / len(reg_views)


# Per-video optimization

def training(dataset, opt, pipe, saving_iterations, visual_iterations):
    spin_iterations = list(range(10000, min(opt.iterations, 30000) + 1, 5000)) if visual_iterations else []
    prepare_output_and_logger(dataset, opt, pipe, saving_iterations, visual_iterations, spin_iterations)
    print("Optimizing with the following parameters:")
    print(dataset.__dict__)
    print(opt.__dict__)

    print("Preparing gaussians and scene...")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_flip=dataset.use_flip)
    gaussians.training_setup(opt)

    print("Preparing deform model...")
    deform_model = DeformModel(scene.v_init, scene.skin_weights, dataset.voxel_deformer_res, use_code=dataset.use_pose_code,
                               is_train=True, optimize_voxel_weights=opt.optimize_voxel_weights, deform_scale=dataset.deform_scale)
    deform_model.train_setting(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    progress_bar = tqdm(range(opt.iterations), desc="Optimization progress")
    losses = defaultdict(AverageMeter)
    viewpoint_cam_list = scene.getTrainCameras()
    cam_idx_list = camera_schedule(len(viewpoint_cam_list), opt)
    # Explicit frame-based noise uses the full clip's time grid, not augmented camera counts.
    time_interval = (scene.frame_time_step if opt.time_noise_std_frames >= 0
                     else 1 / len(viewpoint_cam_list))

    start_time = time.time()
    for iteration in range(1, opt.iterations + 1):
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # data: a frame as (original, flip) camera pair, random background
        viewpoint_cam_dict = viewpoint_cam_list[cam_idx_list[iteration - 1]]
        bg_tensor = torch.rand(3).float().cuda()
        viewpoint_cam = viewpoint_cam_dict["original"]
        viewpoint_flip = viewpoint_cam_dict["flip"]
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
            viewpoint_flip.load2device()
        pose = viewpoint_cam.aux['pose'] # (20, 4, 4)
        gt_image, gt_mask = masked_gt(viewpoint_cam, bg_tensor)
        pose_flip = viewpoint_flip.aux['pose'] # (20, 4, 4)
        gt_image_flip, gt_mask_flip = masked_gt(viewpoint_flip, bg_tensor)

        # deform the canonical Gaussians to this frame
        if iteration < opt.canonical_end:
            gaussians_stages = [gaussians.get_params()]
            stage_names = ["canonical"]
            gaussians_stages_flip = gaussians_stages
        else:
            time_input = viewpoint_cam.fid.unsqueeze(0).cuda()
            time_input_flip = viewpoint_flip.fid.unsqueeze(0).cuda()
            deform_model.update_status(time_interval, iteration, opt)
            gaussians_stages, stage_names = deform_model.train_step(gaussians, pose, time_input, signal=viewpoint_cam.aux['signal'])
            gaussians_stages_flip, _ = deform_model.train_step(gaussians, pose_flip, time_input_flip, signal=viewpoint_flip.aux['signal'])

        # reconstruction of both views (final stage)
        tot_loss = 0.0
        render_pkg_re = render(viewpoint_cam, gaussians_stages[-1], gaussians.active_sh_degree, gaussians.max_sh_degree, pipe, bg_tensor)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        render_pkg_re_flip = render(viewpoint_flip, gaussians_stages_flip[-1], gaussians.active_sh_degree, gaussians.max_sh_degree, pipe, bg_tensor)
        image_flip = render_pkg_re_flip["render"]

        loss_final = loss_func(image, gt_image, opt) + loss_func(image_flip, gt_image_flip, opt)
        losses['rec'].update(loss_final.item() / 2)
        loss_mask = bce_dice_loss(render_pkg_re["alpha"].unsqueeze(0), gt_mask.unsqueeze(0), 1.0) \
            + bce_dice_loss(render_pkg_re_flip["alpha"].unsqueeze(0), gt_mask_flip.unsqueeze(0), 1.0)
        losses['mask'].update(loss_mask.item() / 2)
        tot_loss += loss_final
        tot_loss += loss_mask * opt.lambda_image_reg / 2

        # regularization
        reg_views = [(viewpoint_cam, gaussians_stages, gt_mask, pose)]
        if opt.two_view_reg:
            reg_views.append((viewpoint_flip, gaussians_stages_flip, gt_mask_flip, pose_flip))

        if iteration < opt.pose_sup_until and iteration > opt.canonical_end:
            pose_loss = pose_supervision_loss(deform_model, time_input, reg_views)
            losses["pose"].update(pose_loss.item())
            tot_loss += pose_loss * opt.lambda_pose_sup

        # silhouette of the intermediate (articulated) stage
        render_images = {}
        for i in range(len(gaussians_stages[:-1])):
            render_pkg = render(viewpoint_cam, gaussians_stages[i], gaussians.active_sh_degree, gaussians.max_sh_degree, pipe, bg_tensor)
            render_images[stage_names[i]] = render_pkg["render"]
            render_images[f'{stage_names[i]}_mask'] = render_pkg["alpha"].repeat(3,1,1)
            if opt.use_image_reg:
                loss_i = bce_dice_loss(render_pkg["alpha"].unsqueeze(0), gt_mask.unsqueeze(0), 1.0)
                if opt.two_view_reg:
                    render_pkg_flip = render(viewpoint_flip, gaussians_stages_flip[i], gaussians.active_sh_degree, gaussians.max_sh_degree, pipe, bg_tensor)
                    loss_i = (loss_i + bce_dice_loss(render_pkg_flip["alpha"].unsqueeze(0), gt_mask_flip.unsqueeze(0), 1.0)) / 2
                tot_loss += loss_i * opt.lambda_image_reg
                losses[stage_names[i]].update(loss_i.item())

        losses[stage_names[-1]].update(loss_final.item())
        render_images['flip'] = image_flip
        render_images[stage_names[-1]] = image
        render_images[f'{stage_names[-1]}_mask'] = render_pkg_re["alpha"].repeat(3,1,1)
        render_images['normal'] = render_pkg_re["normal"]

        if opt.use_normal_reg and iteration > opt.normal_reg_from and iteration < opt.normal_reg_until:
            normal_loss = normal_smoothness_loss(opt, reg_views, gaussians, pipe, bg_tensor)
            losses['normal_reg'].update(normal_loss.item())
            tot_loss += normal_loss * opt.lambda_normal_reg

        if opt.use_skin_reg and iteration > opt.canonical_end and iteration < opt.skin_reg_until:
            L_skin = skinning_loss(opt, deform_model, losses)
            tot_loss += L_skin * opt.lambda_skin_reg
            losses['skin_reg'].update(L_skin.item())

        if opt.canonical_end < iteration < opt.compact_until:
            L_compact = compact_loss(deform_model, gaussians_stages, gaussians)
            losses['skin_compact'].update(L_compact.item())
            tot_loss += L_compact * opt.lambda_compact

        if iteration > opt.pose_trans_reg_from and iteration < opt.pose_trans_reg_until and iteration > opt.canonical_end:
            loss_pose_translation = root_translation_loss(deform_model, time_input, reg_views)
            losses["pose_translation"].update(loss_pose_translation.item())
            tot_loss += loss_pose_translation * opt.lambda_pose_trans

        if iteration == opt.freeze_voxel_weights_at:
            deform_model.articulator.voxel_deformer.lbs_voxel_base.requires_grad = False

        tot_loss.backward()
        losses["total"].update(tot_loss.item())

        with torch.no_grad():
            if iteration % 1000 == 0:
                log_progress(progress_bar, losses, gaussians)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                scene.save_deform_ply(gaussians_stages[-1], iteration, save_render_version=True)
                if len(gaussians_stages) > 1:
                    scene.save_deform_ply(gaussians_stages[-2], iteration, save_path=os.path.join(dataset.model_path, "point_cloud", "iteration_{}".format(iteration), "deformed_point_cloud0.ply"), save_render_version=True)
                deform_model.save_weights(dataset.model_path, iteration)

            if iteration in visual_iterations:
                save_visuals(dataset.model_path, iteration, gt_image, render_images, render_pkg_re["depth"], viewpoint_cam,
                             gaussians_stages[-1], gaussians, deform_model, pipe, background)

            if iteration in spin_iterations:
                # Rendering evaluates noisy deformation too; preserve the optimization RNG and status.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    try:
                        for render_skin in [False, True]:
                            render_set_spin(dataset.model_path, "train_spin", iteration, viewpoint_cam_list,
                                            gaussians, pipe, background, deform_model,
                                            save_all_spin=True, use_filter=True, render_skin=render_skin)
                        render_set_spin_yaw_pitch(dataset.model_path, "train_spin", iteration, viewpoint_cam_list,
                                                  gaussians, pipe, background, deform_model,
                                                  use_filter=False, render_skin=False)
                    finally:
                        deform_model.update_status(time_interval, iteration, opt)

            if iteration < opt.iterations:
                densify_and_prune(gaussians, opt, iteration, viewspace_point_tensor, visibility_filter, radii, scene.cameras_extent)
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform_model.optimizer.zero_grad()
                deform_model.update_learning_rate(iteration)

    duration = time.time() - start_time
    print(f"Optimization duration is {duration:.2f} seconds ({duration/60:.2f} minutes)")
    with open(f"{dataset.model_path}/train_time.txt", "w") as f:
        f.write(str(duration))
    return gaussians, deform_model


def densify_and_prune(gaussians, opt, iteration, viewspace_point_tensor, visibility_filter, radii, extent):
    if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter - opt.densify_interval:
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > opt.densify_from_iter:
            if iteration % opt.densify_interval == 0:
                gaussians.densify(opt.densify_grad_threshold, extent)

            if iteration % opt.prune_interval == 0:
                size_threshold = opt.prune_max_screen_size if iteration > opt.opacity_reset_interval else None
                gaussians.prune(opt.prune_opacity_threshold, extent, size_threshold, min_scale_sum=opt.prune_min_scale_sum)

            if iteration % opt.opacity_reset_interval == 0 and iteration <= opt.opacity_reset_until_iter:
                gaussians.reset_opacity()
                print(get_time_prefix() + "Resetting opacity at iteration {}".format(iteration))


def log_progress(progress_bar, losses, gaussians):
    log_str = ""
    for k in losses:
        if losses[k].avg != 0:
            log_str += f"{k}: {losses[k].avg:.{7}f}, "
            losses[k].reset()
    progress_bar.set_postfix({f"N={gaussians.get_xyz.shape[0]}": log_str})
    progress_bar.update(1000)


def save_visuals(model_path, iteration, gt_image, render_images, depth, view, posed, gaussians, deform_model, pipe, background):
    """Save stage comparisons and posed, canonical, and skinning-weight previews."""
    sh = (gaussians.active_sh_degree, gaussians.max_sh_degree)
    vis_render(gt_image, render_images, depth / (depth.max() + 1e-5), f'{model_path}/visual/iter_{iteration}.jpg')
    vis_spin_view(view, posed, *sh, pipe, background, f'{model_path}/visual/spin_{iteration}.gif', n_spin=16)
    canonical = gaussians.get_params()
    vis_canonical_view(view, canonical, *sh, pipe, background, f'{model_path}/visual/can_{iteration}.gif', n_spin=16)
    vis_canonical_view(view, skin_weight_colors(canonical, gaussians, deform_model), *sh, pipe, background,
                       f'{model_path}/visual/skin_{iteration}.gif', n_spin=16)


def prepare_output_and_logger(args, opt, pipe, saving_iterations, visual_iterations, spin_iterations):
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    if visual_iterations:
        os.makedirs(os.path.join(args.model_path, "visual"), exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "training_config.json"), 'w') as config_log_f:
        json.dump({"model": vars(args), "optimization": vars(opt), "pipeline": vars(pipe),
                   "save_iterations": list(saving_iterations),
                   "visual_iterations": list(visual_iterations),
                   "spin_iterations": spin_iterations}, config_log_f, indent=2)
        config_log_f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Optimization parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000, 20000])
    parser.add_argument("--visual_interval", type=int, default=5000,
                        help="visual/ interval; saves one iteration before each boundary; 0 disables visual/ and train_spin/")
    apply_config(parser, sys.argv[1:])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    assert args.model_path, "please set the output folder with -m / --model_path"

    print("Optimizing " + args.model_path)

    set_random_seed(seed=0, deterministic=True)
    visual_iterations = visual_schedule(args.iterations, args.visual_interval, args.canonical_end)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians, deform_model = training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, visual_iterations)

    print("\nOptimization complete.")
