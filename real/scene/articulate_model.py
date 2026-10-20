import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.transforms import quaternion_to_matrix
from utils.graphics_utils import matrix_to_quaternion


class VoxelDeformer(nn.Module):
    def __init__(
        self,
        vtx,
        resolution_dhw=[8, 32, 32],
        short_dim_dhw=0,  # 0 is d, corresponding to z
        long_dim_dhw=1,
    ) -> None:
        super().__init__()
        # Voxel axes (D, H, W) correspond to (z, y, x).
        # Fauna faces +z with +y up, so its short axis is x (W).
        self.resolution_dhw = resolution_dhw
        device = vtx.device
        d, h, w = self.resolution_dhw

        self.register_buffer(
            "ratio",
            torch.Tensor(
                [self.resolution_dhw[long_dim_dhw] / self.resolution_dhw[short_dim_dhw]]
            ).squeeze(),
        )
        self.ratio_dim = -1 - short_dim_dhw
        x_range = (
            (torch.linspace(-1, 1, steps=w, device=device))
            .view(1, 1, 1, w)
            .expand(1, d, h, w)
        )
        y_range = (
            (torch.linspace(-1, 1, steps=h, device=device))
            .view(1, 1, h, 1)
            .expand(1, d, h, w)
        )
        z_range = (
            (torch.linspace(-1, 1, steps=d, device=device))
            .view(1, d, 1, 1)
            .expand(1, d, h, w)
        )
        grid = (
            torch.cat((x_range, y_range, z_range), dim=0)
            .reshape(1, 3, -1)
            .permute(0, 2, 1)
        )

        gt_bbox = torch.cat([vtx.min(dim=1).values, vtx.max(dim=1).values], dim=0).to(
            device
        )
        offset = (gt_bbox[0] + gt_bbox[1])[None, None, :] * 0.5
        self.register_buffer(
            "global_scale", torch.Tensor([1.2]).squeeze()
        )  # from Fast-SNARF
        scale = (gt_bbox[1] - gt_bbox[0]).max() / 2 * self.global_scale

        corner = torch.ones_like(offset[0]) * scale
        corner[0, self.ratio_dim] /= self.ratio
        min_vert = (offset - corner).reshape(1, 3)
        max_vert = (offset + corner).reshape(1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=0)

        self.register_buffer("scale", scale)
        self.register_buffer("offset", offset)

        grid_denorm = self.denormalize(
            grid
        )  # grid_denorm is in the same scale as the canonical body

        self.register_buffer("lbs_voxel_base", torch.empty(0))
        self.register_buffer("grid_denorm", grid_denorm)
        
        d, h, w = self.resolution_dhw  # d-h-w
        scale = self.scale
        ratio = self.ratio

        # edge length of each voxel (xyz order)
        delta_x = 2.0 * scale / (w - 1)
        delta_y = 2.0 * scale / (h - 1)
        delta_z = 2.0 * scale / ((d - 1) * ratio)

        self.register_buffer("voxel_size", torch.tensor([delta_x, delta_y, delta_z]))

        return
    
    def assign_voxel_features(self, vtx, vtx_features, name="lbs_voxel_base", optimizable=False):
        """
        Interpolate vertex skinning weights onto the voxel grid and store them as logits.
        With optimizable=True, register the logits as a parameter; otherwise use a buffer.
        Store the initial logits in a separate buffer named <name>_init.
        """
        weights = (
            self._query_weights_smpl(
                self.grid_denorm,
                smpl_verts=vtx.detach().clone(),
                smpl_weights=vtx_features.detach().clone(),
            )
            .detach()
            .clone()
        )
        
        logits = torch.log(weights.clamp(min=1e-8))

        init_name = name + "_init"
        self.register_buffer(init_name, logits.clone())

        if optimizable:
            if name in self._buffers:
                del self._buffers[name]
            setattr(self, name, nn.Parameter(logits))
        else:
            self.register_buffer(name, logits)

    @property
    def get_voxel_weight(self):
        logits = self.lbs_voxel_base # [1, J, D, H, W]
        return F.softmax(logits, dim=-4)

    def forward(self, xc, mode="bilinear"):
        shape = xc.shape  # ..., 3
        N = 1
        xc = xc.reshape(1, -1, 3)
        w = F.grid_sample(
            self.get_voxel_weight.expand(N, -1, -1, -1, -1), # [1, J, D, H, W]
            self.normalize(xc)[:, :, None, None], # [1, N, 1, 1, 3]
            align_corners=True,
            mode=mode,
            padding_mode="border",
        ) # [1, J, N, 1, 1]
        w = w.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        w = w.reshape(*shape[:-1], -1)
        return w  # [..., J]

    def normalize(self, x):
        x_normalized = x.clone()
        x_normalized -= self.offset
        x_normalized /= self.scale
        x_normalized[..., self.ratio_dim] *= self.ratio
        return x_normalized

    def denormalize(self, x):
        x_denormalized = x.clone()
        x_denormalized[..., self.ratio_dim] /= self.ratio
        x_denormalized *= self.scale
        x_denormalized += self.offset
        return x_denormalized

    def _query_weights_smpl(self, x, smpl_verts, smpl_weights):
        # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
        """Interpolate vertex skinning weights onto the grid, then smooth them.

        Inputs: grid points [1, M, 3], vertices [1, N, 3], weights [1, N, J].
        Returns voxel weights [1, J, D, H, W].
        """
        dist, idx, _ = knn_points(x, smpl_verts.detach(), K=30)  # [B, M, K]
        dist = dist.sqrt().clamp_(0.0001, 1.0)  # avoid division by zero for grid points on a vertex
        weights = smpl_weights[0, idx]  # [B, M, K, J]

        ws = 1.0 / dist  # [B, M, K]
        ws = ws / ws.sum(-1, keepdim=True)
        weights = (ws[..., None] * weights).sum(-2)  # [B, M, J]

        c = smpl_weights.shape[-1] # J
        d, h, w = self.resolution_dhw
        weights = weights.permute(0, 2, 1).reshape(1, c, d, h, w) # [1, J, D, H, W]
        for _ in range(30):
            mean = (
                weights[:, :, 2:, 1:-1, 1:-1]
                + weights[:, :, :-2, 1:-1, 1:-1]
                + weights[:, :, 1:-1, 2:, 1:-1]
                + weights[:, :, 1:-1, :-2, 1:-1]
                + weights[:, :, 1:-1, 1:-1, 2:]
                + weights[:, :, 1:-1, 1:-1, :-2]
            ) / 6.0
            weights[:, :, 1:-1, 1:-1, 1:-1] = (
                weights[:, :, 1:-1, 1:-1, 1:-1] - mean
            ) * 0.7 + mean
            sums = weights.sum(1, keepdim=True)
            weights = weights / sums
        return weights.detach()


class ArticulateModel(nn.Module):
    def __init__(self, verts, voxel_deformer_res=64):
        super().__init__()
        
        self.name = "fauna"
        self.compact_alpha = 2.0

        self.voxel_deformer = VoxelDeformer(
            vtx=verts.unsqueeze(0),
            resolution_dhw=[
                voxel_deformer_res,
                voxel_deformer_res,
                voxel_deformer_res,
            ],
            short_dim_dhw=2,
            long_dim_dhw=0,
        )
        
    def assign_voxel_weights(self, verts, verts_weights, optimizable=False):
        # verts: N, 3. If optimizable=True, voxel weights become nn.Parameter and can be optimized.
        self.voxel_deformer.assign_voxel_features(
            verts.unsqueeze(0), verts_weights, optimizable=optimizable
        )
        self.num_joints = verts_weights.shape[-1]
                
        
    def forward(self, xyz_canonical, rot_canonical, pose):
        xyz_can = xyz_canonical.unsqueeze(0)  # B, N, 3
        rot_can = quaternion_to_matrix(rot_canonical).unsqueeze(0)  # B, N, 3, 3
        pose = pose.unsqueeze(0) # B, J, 4, 4
        
        W = self.voxel_deformer(xyz_can)  # B,N,J
        self.w_prob = W  # cached for get_sparsity_loss / compute_compact_loss
        T = torch.einsum("bnk, bkrc -> bnrc", W, pose) # per-Gaussian transform (B, N, 4, 4)
        
        
        R, t = T[:, :, :3, :3], T[:, :, :3, 3]  # B,N,3,3; B,N,3
        mu = torch.einsum("bnij,bnj->bni", R, xyz_can) + t  # B,N,3
        frame = torch.einsum("bnij,bnjk->bnik", R, rot_can)  # B,N,3,3
        mu = mu.squeeze(0)
        rot = matrix_to_quaternion(frame[0])  # N,4
        rot = torch.nn.functional.normalize(rot)
        
        return mu, rot


    def get_tv_loss(self):
        grid_w_prob = self.voxel_deformer.get_voxel_weight  # [1, J, D, H, W]
        
        diff_x = grid_w_prob[:, :, :, :, 1:] - grid_w_prob[:, :, :, :, :-1]
        diff_y = grid_w_prob[:, :, :, 1:, :] - grid_w_prob[:, :, :, :-1, :]
        diff_z = grid_w_prob[:, :, 1:, :, :] - grid_w_prob[:, :, :-1, :, :]
        
        delta = 0.05
        
        # Huber loss implemented by hand (avoids the zeros_like allocation)
        def fast_huber(diff, delta):
            abs_diff = torch.abs(diff)
            # L2 (0.5 * x^2) below delta, L1 (delta * x - 0.5 * delta^2) above
            return torch.where(abs_diff < delta, 
                               0.5 * diff ** 2, 
                               delta * abs_diff - 0.5 * delta ** 2).mean()

        tv_x = fast_huber(diff_x, delta)
        tv_y = fast_huber(diff_y, delta)
        tv_z = fast_huber(diff_z, delta)
        
        return (tv_x + tv_y + tv_z) / 3.0
        
    def get_sparsity_loss(self):
        """
        sparsity (low entropy) of the skinning weights of the current Gaussians.
        Uses self.w_prob [1, N, J] cached by the last forward() call.
        """
        # Shannon entropy
        # add 1e-8 to avoid log(0) = NaN
        entropy = -torch.sum(self.w_prob * torch.log(self.w_prob + 1e-8), dim=-1) # [1, N]
        
        return entropy.mean()
    

    def compute_compact_loss(self, x_can):
        """per-bone weighted covariance of x_can (weights: self.w_prob from the last forward()); penalizes the cross-section"""
        device = x_can.device
        W_prob = self.w_prob.squeeze(0).detach()  # [N, J]
        W_sum = W_prob.sum(dim=0) + 1e-8

        # centroid weighted by W_prob
        mu = (x_can.t() @ W_prob / W_sum).t()  # [J, 3]

        # offsets [N, J, 3]
        diff = x_can.unsqueeze(1) - mu.unsqueeze(0)

        # weighted covariance [J, 3, 3]
        diff_w = diff * W_prob.unsqueeze(-1)  # [N, J, 3]
        cov = torch.einsum('njc,njd->jcd', diff_w, diff) / W_sum.unsqueeze(-1).unsqueeze(-1)
        cov = cov + torch.eye(3, device=device).unsqueeze(0) * 1e-5

        # adaptive weight: joints with larger W_sum are penalized less
        w_sum_norm = W_sum / W_sum.max()
        joint_weight = torch.exp(-self.compact_alpha * w_sum_norm)
    
        # eigenvalues [J, 3], ascending
        eigvals = torch.linalg.eigvalsh(cov)  # [J, 3]

        # only penalize the two smallest eigenvalues (cross-section directions)
        cross_section_spread = eigvals[:, :2].sum(dim=-1)  # [J]

        return (cross_section_spread * joint_weight).mean()
