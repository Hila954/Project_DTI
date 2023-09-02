import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.flow_utils import norm_lms, resize_flow_tensor

class KptsLoss(nn.modules.Module):
    def __init__(self, args):
        super(KptsLoss, self).__init__()
        self.shape = args.orig_shape

    def forward(self, pred_flow, kpts, vox_dim, mode='l1'):
        lms_fixed = torch.tensor(kpts['lms_fixed'], dtype=torch.float, device=pred_flow.device).reshape(1,3,1,1,-1)
        lms_moving = torch.tensor(kpts['lms_moving'], dtype=torch.float, device=pred_flow.device).reshape(1,3,1,1,-1)
        vox_dim = vox_dim.reshape(pred_flow.shape[0],3,1,1,1)

        #B, _, H, W, D = pred_flow.size()
        #pred_flow = torch.flip(pred_flow, [1])
        #base_grid = mesh_grid(B, H, W, D).type_as(pred_flow)  # B2HW

        #v_grid = norm_grid(base_grid + flow12)  # BHW2
        flow12 = resize_flow_tensor(pred_flow[:, :3].squeeze(0),shape=self.shape)
        flow21 = resize_flow_tensor(pred_flow[:, 3:].squeeze(0),shape=self.shape)
        #flow12 = torch.flip(flow12, [1])
        #flow21 = torch.flip(flow21, [1])

        grid_fixed = norm_lms(lms_fixed, flow12.shape)  # BHW2
        grid_moving = norm_lms(lms_moving, flow21.shape)  # BHW2

        disps_12 = nn.functional.grid_sample(
            flow12, grid_fixed, mode='bilinear', padding_mode='border', align_corners=True)

        disps_21 = nn.functional.grid_sample(
            flow21, grid_moving, mode='bilinear', padding_mode='border', align_corners=True)

        lms_fixed_disp = lms_fixed + disps_12
        lms_moving_disp = lms_moving + disps_21

        if mode == 'l1':
            loss = ((vox_dim * (lms_fixed_disp - lms_moving)).abs().mean() + \
                (vox_dim * (lms_moving_disp - lms_fixed)).abs().mean()) / 2
        elif mode == 'l2':
            loss = (((vox_dim * (lms_fixed_disp - lms_moving))**2).mean() + \
                ((vox_dim * (lms_moving_disp - lms_fixed))**2).mean()) / 2

        return loss

