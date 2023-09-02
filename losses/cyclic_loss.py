import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
import numpy as np

from utils.flow_utils import flow_warp

class CyclicLoss(nn.modules.Module):
    def __init__(self, args):
        super(CyclicLoss, self).__init__()
        self.cyc_w_scales = args.cyc_w_scales
        self.w_bk = args.w_bk

    def forward(self, output, pyramid_occu_masks, vox_dim):
        pyramid_flows = output
        pyramid_cyc_losses = []
        for i, (flow,occu_masks) in enumerate(zip(pyramid_flows,pyramid_occu_masks)):

            if self.cyc_w_scales[i] > 0:
                
                flow12 = flow[:, :3]
                flow21 = flow[:, 3:]
                occu1, occu2 = occu_masks
                flow21_warped = flow_warp(flow21, flow12)
                flow12_warped = flow_warp(flow12, flow21)
                spacing = vox_dim.reshape(flow.shape[0],3,1,1,1) / 2**i

                loss_cyc = ((spacing * (flow21_warped * occu1 + flow12 * occu1))**2).mean() / occu1.mean() + \
                            ((spacing * (flow12_warped * occu2 + flow21 * occu2))**2).mean() / occu2.mean()
                loss_cyc /= 2

                if np.isnan(loss_cyc.item()):
                    print("nan")
            else:
                loss_cyc = 0.0

            pyramid_cyc_losses.append(loss_cyc)

        pyramid_cyc_losses = [l * w for l, w in zip(pyramid_cyc_losses, self.cyc_w_scales)]
        loss_cyc = sum(pyramid_cyc_losses)
        return loss_cyc
