import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class JointSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):  # x: (N, D)
        N, D = x.shape

        qkv = self.to_qkv(x)  # (N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (N, D)

        # split into heads: (heads, N, head_dim)
        q = q.view(N, self.heads, self.head_dim).transpose(0, 1)  # (H, N, d)
        k = k.view(N, self.heads, self.head_dim).transpose(0, 1)  # (H, N, d)
        v = v.view(N, self.heads, self.head_dim).transpose(0, 1)  # (H, N, d)

        # Attention score: (H, N, N)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # aggregate: (H, N, d)
        out = attn_weights @ v

        # merge heads back to (N, D)
        out = out.transpose(0, 1).contiguous().view(N, D)
        return self.out_proj(out)


class JointPoseNetSmall(nn.Module):
    def __init__(self, time_dim, joint_dim, hidden_dim, num_joints, code_dim, heads=2):
        super().__init__()
        self.joint_embed = nn.Parameter(torch.randn(num_joints, joint_dim))
        self.input_mlp = nn.Sequential(
            nn.Linear(time_dim + joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.attn = JointSelfAttention(hidden_dim, heads=heads)
        self.out_proj = nn.Linear(hidden_dim + time_dim + joint_dim, hidden_dim)
        self.fc_rot = nn.Linear(hidden_dim, 4)
        self.fc_trans = nn.Linear(hidden_dim, 3)
        self.fc_code = nn.Linear(hidden_dim, code_dim)
        self._init_pose_outputs_identity()

    def _init_pose_outputs_identity(self):
        """Initialize pose as identity: rotation = unit quaternion, translation = 0."""
        nn.init.zeros_(self.fc_rot.weight)
        nn.init.zeros_(self.fc_rot.bias)
        self.fc_rot.bias.data[0] = 1.0  # (1,0,0,0) -> unit quaternion after normalization
        nn.init.zeros_(self.fc_trans.weight)
        nn.init.zeros_(self.fc_trans.bias)

    def forward(self, time_embedding):  # (1, D)
        N = self.joint_embed.shape[0]
        time = time_embedding.expand(N, -1)  # (N, D)
        x = torch.cat([time, F.normalize(self.joint_embed)], dim=-1)  # (N, D+J)
        feat = self.input_mlp(x)                         # (N, H)
        feat = self.attn(feat)                           # (N, H)
        feat = F.relu(feat)
        feat = self.out_proj(torch.cat([feat, time, self.joint_embed], dim=-1))  # (N, H)
        feat = F.relu(feat)

        qua = F.normalize(self.fc_rot(feat), dim=-1)   # (N, 4)
        trans = self.fc_trans(feat)  # (N, 3)
        pose_code = self.fc_code(feat)  # (N, code_dim)
        return qua, trans, pose_code


def cross_attention(query, key, value, num_heads=4):
    dim = query.shape[-1]
    head_dim = dim // num_heads
    scale = head_dim ** -0.5

    q = query.view(-1, num_heads, head_dim).transpose(0, 1)  # (H, N_q, d)
    k = key.view(-1, num_heads, head_dim).transpose(0, 1)    # (H, N_k, d)
    v = value.view(-1, num_heads, head_dim).transpose(0, 1)  # (H, N_k, d)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (H, N_q, N_k)
    attn_weights = F.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_weights, v)  # (H, N_q, d)
    out = out.transpose(0, 1).contiguous().view(-1, dim)  # (N_q, D)
    return out
    

class DeformNetWithJointAttention(nn.Module):
    def __init__(self, pose_dim, hidden_dim=128, mlp_layers=4, num_heads=4, input_dim=9):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.k_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
        )
        self.v_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
        )

        self.query_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.linear_pose = nn.ModuleList(
            [nn.Linear(hidden_dim + input_dim, hidden_dim)] +
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(1, mlp_layers - 1)])

        self.out_x = nn.Linear(hidden_dim, 3)
        self.out_q = nn.Linear(hidden_dim, 4)
        self.out_s = nn.Linear(hidden_dim, 3)


    def forward(self, xqs, pose_code):

        key = self.k_proj(pose_code)  # (N_joint, D)
        value = self.v_proj(pose_code)  # (N_joint, D)

        query = self.query_proj(xqs)       # (N_gauss, D)
        
        feat = cross_attention(query, key, value, num_heads=self.num_heads)  # (N_gauss, D)

        h = self.linear_pose[0](torch.cat([feat, xqs], dim=-1))  # (N_gauss, D)
        for layer in self.linear_pose[1:]:
            h = F.relu(layer(h))

        dx = self.out_x(h)  # (N_gauss, 3)
        dx = F.tanh(dx) * 0.5  # ensure the displacement is not too large
        dq = self.out_q(h)  # (N_gauss, 4)
        dq = F.normalize(dq, dim=-1)
        ds = self.out_s(h)  # (N_gauss, 3), log scale
        

        return dx, dq, ds

class DeformPoseTimeNetwork(nn.Module):
    """Predict bone poses and pose codes, then Gaussian position offsets, rotations, and scale logits."""
    def __init__(self, multires=10, J=20, joint_dim=8, pose_hidden_dim=128, deform_hidden_dim=256, code_size=64, use_code=True):
        super(DeformPoseTimeNetwork, self).__init__()
        self.t_multires = multires
        self.J = J
        self.code_size = code_size
        self.joint_dim = joint_dim
        self.pose_hidden_dim = pose_hidden_dim
        self.deform_hidden_dim = deform_hidden_dim
        self.use_code = use_code

        self.embed_time_fn, time_input_ch = get_embedder(multires, 1)
        self.embed_x_fn, x_input_ch = get_embedder(multires, 3)
        
        
        # time embedding -> per-bone pose + pose code
        self.estimate_pose = JointPoseNetSmall(time_input_ch+1, joint_dim, pose_hidden_dim, J, self.code_size, heads=2)
        self.joint_anchor = self.estimate_pose.joint_embed
        
        # per-Gaussian non-rigid deformation
        input_dim = 7 + x_input_ch + joint_dim
        self.deform_net = DeformNetWithJointAttention(code_size, deform_hidden_dim, mlp_layers=3, num_heads=4, input_dim=input_dim)
                
    
    def compose_transform(self, pose_qua, pose_t):
        pose_rot = quaternion_to_matrix(pose_qua)  # (J, 3, 3)
        bottom_row = torch.tensor([0, 0, 0, 1], device=pose_t.device).view(1, 1, 4).repeat(pose_t.size(0), 1, 1)
        pose_upper = torch.cat([pose_rot, pose_t.unsqueeze(-1)], dim=2)  # (J, 3, 4)
        pose = torch.cat([pose_upper, bottom_row], dim=1)  # (J, 4, 4)
        return pose
    
    def estimate_pose_forward(self, time, signal=None):
        # time: (1, 1)
        # signal: [m, v], see scene.dataset_readers.make_camera
        # returns quaternions (J, 4), translations (J, 3), pose codes (J, code_size)
        if signal is None:
            signal = torch.tensor([1., 1.]).to(time.device).float()
        m, v = signal
        h = self.embed_time_fn(time * m)
        pred_qua, pred_trans, pred_code = self.estimate_pose(torch.cat([h, v.reshape(1,-1)], dim=-1))
        if not self.use_code:
            pred_code = torch.zeros_like(pred_code)
        return pred_qua, pred_trans, pred_code
    
    def refine_deformation_forward(self, x, q, s, pose_code, W_prob):
        # x: (N, 3), q: (N, 4), s: (N, 3)
        # pose_code: (J, code_size)
        # W_prob: (N, J)
        x_embed = self.embed_x_fn(x)
        anchor_features = self.attach_joint_anchor(W_prob)
        xqs = torch.cat([x_embed, q, s, anchor_features], dim=-1)  # (N, x_input_ch + 7 + joint_dim)
        d_x, d_q, d_s = self.deform_net(xqs, pose_code)
        return d_x, d_q, d_s
    
    def attach_joint_anchor(self, W_prob):
        # W_prob: (N, J)
        anchor_features = W_prob @ self.joint_anchor # (N,J) @ (J, joint_dim) -> (N, joint_dim)
        return anchor_features
