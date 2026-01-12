import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionCompensation(nn.Module):
    def __init__(self, bev_h, bev_w, pc_range):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        x = torch.linspace(-1, 1, bev_w)
        y = torch.linspace(-1, 1, bev_h)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('base_grid', torch.stack([xx, yy], dim=-1))

    def forward(self, bev_features, ego_motion):
        B, C, H, W = bev_features.shape
        device = bev_features.device
        grid = self.base_grid.unsqueeze(0).expand(B, -1, -1, -1).to(device)
        rot = ego_motion[:, :2, :2]
        trans = ego_motion[:, :2, 3:4]
        grid_flat = grid.view(B, -1, 2)
        new_grid = torch.bmm(grid_flat, rot.transpose(1, 2))
        scale_x = (self.pc_range[3] - self.pc_range[0]) / 2
        scale_y = (self.pc_range[4] - self.pc_range[1]) / 2
        trans_norm = trans.squeeze(-1) / torch.tensor([scale_x, scale_y], device=device)
        new_grid = (new_grid + trans_norm.unsqueeze(1)).view(B, H, W, 2)
        return F.grid_sample(bev_features, new_grid, mode='bilinear', padding_mode='zeros', align_corners=False)

class LightweightTemporalFusion(nn.Module):
    def __init__(self, dim, num_frames, bev_h, bev_w, pc_range):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.motion_comp = MotionCompensation(bev_h, bev_w, pc_range)
        self.fuse = nn.Conv2d(dim * 2, dim, 1)
        self.gate = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())
        self.history = []
        self.history_poses = []

    def reset(self):
        self.history = []
        self.history_poses = []

    def forward(self, current_bev, ego_motion=None, current_pose=None):
        if len(self.history) == 0:
            self._update_history(current_bev, current_pose)
            return current_bev
        B, C, H, W = current_bev.shape
        device = current_bev.device
        hist_bev, hist_pose = self.history[-1], self.history_poses[-1]
        if current_pose is not None and hist_pose is not None:
            rel_pose = torch.bmm(torch.inverse(current_pose), hist_pose)
        else:
            rel_pose = ego_motion if ego_motion is not None else torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
        aligned = self.motion_comp(hist_bev.to(device), rel_pose)
        concat = torch.cat([current_bev, aligned], dim=1)
        fused = self.fuse(concat)
        gate = self.gate(concat)
        output = gate * fused + (1 - gate) * current_bev
        self._update_history(current_bev, current_pose)
        return output

    def _update_history(self, bev, pose):
        self.history.append(bev.detach().clone())
        self.history_poses.append(pose.detach().clone() if pose is not None else None)
        if len(self.history) > self.num_frames - 1:
            self.history.pop(0)
            self.history_poses.pop(0)
