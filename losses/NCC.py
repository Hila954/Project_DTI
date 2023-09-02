import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np


"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims
        sum_NCC = 0
        weight_win_size = self.win_raw
        num_channels = I.shape[1]
        splitted_I = np.split(I, I.shape[1], axis=1)
        splitted_J = np.split(J, J.shape[1], axis=1)

        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d
        del I, J
        torch.cuda.empty_cache()
        for channel in range(num_channels):
            I = splitted_I[channel]
            J = splitted_J[channel]
            # compute CC squares
            I2 = I*I
            J2 = J*J
            IJ = I*J

            # compute filters
            # compute local sums via convolution
            I_sum = conv_fn(I, weight, padding=int(win_size/2))
            J_sum = conv_fn(J, weight, padding=int(win_size/2))
            I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
            J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
            IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))
            del I, J
            torch.cuda.empty_cache()
                # compute cross correlation
            win_size = np.prod(self.win)
            u_I = I_sum/win_size
            u_J = J_sum/win_size

            cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
            del I2, J2, IJ, I_sum, u_I, IJ_sum, I2_sum
            torch.cuda.empty_cache()
            J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

            del J_sum, J2_sum, u_J
            torch.cuda.empty_cache()
            cc = cross * cross / (I_var * J_var + self.eps)
            if channel == 0:
                sum_NCC += torch.mean(cc)
            else:
                sum_NCC += torch.mean(cc).item()
            del cross, I_var, J_var, cc
            torch.cuda.empty_cache()

        avg_NCC_along_channels = sum_NCC/num_channels
        

        # return negative cc.
        return -1.0 * torch.mean(avg_NCC_along_channels)


    # def forward(self, I, J):
    #     ndims = 3
    #     win_size = self.win_raw
    #     self.win = [self.win_raw] * ndims

    #     weight_win_size = self.win_raw
    #     weight = torch.ones((1, I.shape[1], weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
    #     conv_fn = F.conv3d

    #     # compute CC squares
    #     I2 = I*I
    #     J2 = J*J
    #     IJ = I*J

    #     # compute filters
    #     # compute local sums via convolution
    #     I_sum = conv_fn(I, weight, padding=int(win_size/2))
    #     J_sum = conv_fn(J, weight, padding=int(win_size/2))
    #     I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
    #     J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
    #     IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

    #     # compute cross correlation
    #     win_size = np.prod(self.win)
    #     u_I = I_sum/win_size
    #     u_J = J_sum/win_size

    #     cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    #     I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    #     J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

    #     cc = cross * cross / (I_var * J_var + self.eps)

    #     # return negative cc.
    #     return -1.0 * torch.mean(cc)
