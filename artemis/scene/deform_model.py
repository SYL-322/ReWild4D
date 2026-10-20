import torch
import torch.nn as nn
from .deforms import DeformPoseTimeNetwork
from .articulate_model import ArticulateModel
from .gaussian_model import GaussianParams
import os
from utils.general_utils import get_expon_lr_func, get_linear_noise_func


NUM_JOINTS = 20  # bones of the Fauna skeleton


def _get_submodule_and_attr(module, key):
    """key like 'voxel_deformer.lbs_voxel_base' -> (voxel_deformer submodule, 'lbs_voxel_base')"""
    parts = key.split(".")
    attr_name = parts[-1]
    for p in parts[:-1]:
        module = getattr(module, p)
    return module, attr_name


class DeformModel:
    def __init__(self, v_init, skin_weights, voxel_deformer_res, use_code=True, is_train=False, optimize_voxel_weights=False):
        print(f"Prepare {self.__class__.__name__} model")
        self.optimize_voxel_weights = optimize_voxel_weights
        self.articulator = ArticulateModel(v_init, voxel_deformer_res)
        self.v_init = v_init
        self.skin_weights = skin_weights
        if is_train:
            self.articulator.assign_voxel_weights(
                v_init, skin_weights, optimizable=optimize_voxel_weights
            )
        self.articulator.to('cuda')
        for param in self.articulator.parameters():
            param.requires_grad = False
        # When optimizing voxel weights, keep lbs_voxel_base trainable
        if optimize_voxel_weights and hasattr(self.articulator.voxel_deformer, 'lbs_voxel_base'):
            w = self.articulator.voxel_deformer.lbs_voxel_base
            if isinstance(w, nn.Parameter):
                w.requires_grad = True
        print("Articulator model loaded")
        
        self.deform = DeformPoseTimeNetwork(J=NUM_JOINTS, use_code=use_code).cuda()
        self.optimizer = None
        print(f"{self.deform.__class__.__name__} deform model loaded")

    def update_status(self, time_interval, iteration=None, opt=None):
        """training (time_interval > 0): decaying time noise; test (time_interval = 0): no noise"""
        if time_interval > 0:
            self.time_noise = self.time_noise_smooth_term(iteration) * time_interval
            self.use_deform = iteration >= opt.warm_up
        else:
            self.time_noise = 0.0
            self.use_deform = True

    def train_step(self, gaussians, time, signal=None):
        '''
        Deform the canonical Gaussians to one frame, with the bone transforms predicted for this time.
        time: (1, 1)
        signal: [m, v], see scene.dataset_readers.make_camera
        returns [articulated, deformed] GaussianParams (only articulated during warm-up) and their names
        '''
        canonical = gaussians.get_params()
        
        time_input = time + torch.randn(1, 1, device='cuda') * self.time_noise
        pred_qua, pred_trans, pose_code = self.deform.estimate_pose_forward(time_input, signal=signal)
        pred_pose = self.deform.compose_transform(pred_qua, pred_trans) # (J, 4, 4)
        posed_xyz, posed_rotation = self.articulator(canonical.xyz, canonical.rotation, pred_pose)
        W_prob = self.articulator.w_prob.squeeze(0)  # [N, J]
        articulated = canonical._replace(xyz=posed_xyz, rotation=posed_rotation)
        if not self.use_deform:
            return [articulated], ["refine_articulated"]
        
        # non-rigid refinement on top of the articulated shape (the skinning weights get gradients from it as well)
        d_xyz, d_rotation, d_scaling = self.deform.refine_deformation_forward(
            posed_xyz.detach(), posed_rotation.detach(), gaussians._scaling.detach(), pose_code, W_prob)
        deformed_rotation = gaussians.rotation_activation(d_rotation)
        deformed_scales = gaussians.scaling_activation(d_scaling)
        deformed = canonical._replace(xyz=posed_xyz + d_xyz, scaling=deformed_scales, rotation=deformed_rotation)
        # The articulated stage shares rotation and scale, with frozen opacity and color.
        articulated = GaussianParams(posed_xyz, deformed_scales, deformed_rotation, canonical.opacity.detach(), canonical.features.detach())
        return [articulated, deformed], ["refine_articulated", "deformed"]

    def train_setting(self, training_args):
        self.time_noise_smooth_term = get_linear_noise_func(lr_init=0.5, lr_final=1e-15, lr_delay_steps=0, lr_delay_mult=0.01, max_steps=training_args.time_noise_steps)
        
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.deform_lr_init,
             "name": "deform"}
        ]
        voxel_lr = getattr(training_args, 'voxel_weight_lr', training_args.deform_lr_init * 0.1)
        if self.optimize_voxel_weights and hasattr(self.articulator.voxel_deformer, 'lbs_voxel_base'):
            w = self.articulator.voxel_deformer.lbs_voxel_base
            if isinstance(w, nn.Parameter):
                l.append({'params': [w], 'lr': voxel_lr, "name": "voxel_weight"})
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.deform_lr_init,
                                                       lr_final=training_args.deform_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)
        self.voxel_weight_scheduler_args = get_expon_lr_func(
            lr_init=voxel_lr, lr_final=voxel_lr * 0.1,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps,
        ) if self.optimize_voxel_weights else None

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))
        torch.save(self.articulator.state_dict(), os.path.join(out_weights_path, 'articulator.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = "best"
        else:
            loaded_iter = iteration
        base_path = os.path.join(model_path, "deform/iteration_{}".format(loaded_iter))
        self.deform.load_state_dict(torch.load(os.path.join(base_path, 'deform.pth')))
        articulator_path = os.path.join(base_path, 'articulator.pth')
        if os.path.isfile(articulator_path):
            ckpt = torch.load(articulator_path)
            model_state = self.articulator.state_dict()
            # load params with matching shapes; at eval lbs_voxel_base starts as [0] and is assigned from the ckpt
            to_load = {}
            for k, v in ckpt.items():
                if k in model_state and model_state[k].shape == v.shape:
                    to_load[k] = v
                elif k in model_state and model_state[k].numel() == 0 and v.numel() > 0:
                    # placeholder buffer (e.g. lbs_voxel_base [0]): register it with the checkpoint value
                    submod, attr_name = _get_submodule_and_attr(self.articulator, k)
                    dev = next(self.articulator.buffers()).device
                    submod.register_buffer(attr_name, v.to(dev))
            self.articulator.load_state_dict(to_load, strict=False)
        else:
            print(f"[Warning] articulator weights not found at {articulator_path}, using the Fauna skinning weights")
            self.articulator.assign_voxel_weights(self.v_init.cuda(), self.skin_weights.cuda())

    def update_learning_rate(self, iteration):
        lr = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "voxel_weight" and self.voxel_weight_scheduler_args is not None:
                param_group['lr'] = self.voxel_weight_scheduler_args(iteration)
        return lr
