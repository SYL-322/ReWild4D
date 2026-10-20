"""Real-video KID/FVD protocols; consumes prepared frames and generated novel-view images."""
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import piq
from ..protocols.constants import N_SPIN
from ..protocols.real_frames import load_robust_frames, matched_novel_paths
from .fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained

device = 'cuda'
_i3d = None
_kid = None

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_gt_frames(file_names, data_dir, frame_resolution=224):
    """input frames with white background, [time, channel, h, w]"""
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    frame_list = []
    for name in file_names:
        img = Image.open(os.path.join(img_dir, name)).convert("RGB").resize((frame_resolution, frame_resolution))
        mask = Image.open(os.path.join(mask_dir, name)).convert("L").resize((frame_resolution, frame_resolution))
        img_np = np.array(img)
        img_np[np.array(mask) < 125] = 255
        frame_list.append(torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
    return torch.cat(frame_list, dim=0)

def load_video(paths, frame_resolution=224):
    frames = [torch.from_numpy(np.array(Image.open(f).resize((frame_resolution, frame_resolution)))).permute(2, 0, 1).unsqueeze(0).float() / 255.0 for f in paths]
    return torch.cat(frames, dim=0).unsqueeze(0).to(device)  # [1, time, channel, h, w]

def calculate_fvd(videos1, videos2):
    """videos: [batch_size, timestamps, channel, h, w], at least 10 timestamps"""
    assert videos1.shape == videos2.shape
    # BTCHW -> BCTHW
    videos1 = videos1.permute(0, 2, 1, 3, 4)
    videos2 = videos2.permute(0, 2, 1, 3, 4)
    assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"
    feats1 = get_fvd_feats(videos1, i3d=_i3d, device=device)
    feats2 = get_fvd_feats(videos2, i3d=_i3d, device=device)
    return frechet_distance(feats1, feats2)

def eval_fvd(data_path, novel_view_dir, frame_resolution=224, seed=0):
    """
    fvd_diag: video whose view rotates with time (view t % N_SPIN at frame t)
    fvd_f:    mean over 4 fixed-view videos (0, 90, 180, 270 degrees)
    """
    global _i3d
    frames, selected_frame_ids = load_robust_frames(data_path)
    selected_frame_names = [frames[i]['file_path'] for i in selected_frame_ids]
    frame_names = [os.path.splitext(os.path.basename(name))[0] for name in selected_frame_names]
    paths = matched_novel_paths(novel_view_dir, frame_names, N_SPIN)
    if _i3d is None:
        _i3d = load_i3d_pretrained(device=device)
    set_random_seed(seed)  # after creating the network
    gt_video = get_gt_frames(selected_frame_names, data_path, frame_resolution).unsqueeze(0).to(device)  # [1, time, channel, h, w]

    time_total = len(frame_names)
    diag_paths = [paths[t][t % N_SPIN] for t in range(time_total)]
    fvd_diag = calculate_fvd(load_video(diag_paths, frame_resolution), gt_video)

    fvd_f_list = []
    for vid in [0, N_SPIN // 4, N_SPIN // 2, 3 * N_SPIN // 4]:
        f_paths = [paths[t][vid] for t in range(time_total)]
        fvd_f_list.append(calculate_fvd(load_video(f_paths, frame_resolution), gt_video))

    return {"fvd_diag": fvd_diag, "fvd_f": np.mean(fvd_f_list)}

class ImageFileDataset(Dataset):
    def __init__(self, image_paths, image_size=512):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.image_paths[idx]).convert("RGB")
        return {'images': self.transform(rgb)}

class ImageMaskFileDataset(Dataset):
    """masked input frames on a constant background"""
    def __init__(self, image_paths, mask_paths, image_size=512, bg_color=(255, 255, 255)):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.bg_color = bg_color
        self.resolution = (image_size, image_size)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.image_paths[idx]).convert("RGB").resize(self.resolution, Image.LANCZOS)
        mask = Image.open(self.mask_paths[idx]).convert("L").resize(self.resolution, Image.LANCZOS)
        rgb_np = np.array(rgb)
        rgb_np[np.array(mask) < 125] = self.bg_color
        return {'images': self.transform(rgb_np)}

def eval_kid(data_path, novel_view_dir, eval_n_spin=16, image_size=512, batch_size=8, seed=0):
    """KID between robust input frames and eval_n_spin novel views of retained validation frames."""
    global _kid
    if not isinstance(eval_n_spin, int) or not 0 < eval_n_spin <= N_SPIN or N_SPIN % eval_n_spin:
        raise ValueError(f'eval_n_spin must be a positive divisor of {N_SPIN}')
    frames, selected_frame_ids = load_robust_frames(data_path)
    selected_frame_names = [frames[i]['file_path'] for i in selected_frame_ids]
    frame_names = [os.path.splitext(os.path.basename(name))[0] for name in selected_frame_names]
    paths = matched_novel_paths(novel_view_dir, frame_names, N_SPIN)
    select_views = [i * (N_SPIN // eval_n_spin) for i in range(eval_n_spin)]
    val_indices = [j for j, i in enumerate(selected_frame_ids) if frames[i]['is_val']]
    fake_paths = [paths[j][vid] for vid in select_views for j in val_indices]
    if not fake_paths:
        print(f"skip KID: no validation frames in {data_path}")
        return None
    if _kid is None:
        from piq.feature_extractors import InceptionV3
        _kid = (InceptionV3(), piq.KID().to(device))
    feat_extractor, kid_metric = _kid
    set_random_seed(seed)  # after creating the networks, KID samples random subsets


    # real: input frames
    img_paths = [os.path.join(data_path, "images", name) for name in selected_frame_names]
    mask_paths = [os.path.join(data_path, "masks", name) for name in selected_frame_names]
    real_dl = DataLoader(ImageMaskFileDataset(img_paths, mask_paths, image_size=image_size), batch_size=batch_size, shuffle=False)

    # fake: novel views of the validation frames
    fake_dl = DataLoader(ImageFileDataset(fake_paths, image_size=image_size), batch_size=batch_size, shuffle=False)

    real_feats = kid_metric.compute_feats(real_dl, feat_extractor, device=device)
    fake_feats = kid_metric.compute_feats(fake_dl, feat_extractor, device=device)
    return kid_metric(real_feats, fake_feats).item()
