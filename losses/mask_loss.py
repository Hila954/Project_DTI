import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math

from utils.flow_utils import flow_warp

class MaskWarpLoss(nn.modules.Module):
    def __init__(self, args):
        super(MaskWarpLoss, self).__init__()
        self.mwl_w_scales = args.mwl_w_scales
        self.w_bk = args.w_bk
        self.mwl_apply_gaussian = args.mwl_apply_gaussian
        self.mwl_use_occ_masks = args.mwl_use_occ_masks
        if self.mwl_apply_gaussian:
            self.gaussian_smoothing = GaussianSmoothing(channels=1,kernel_size=args.mwl_gaus_kernel,sigma=args.mwl_gaus_sigma,dim=3)

    def loss_func(self,msk,msk_recons,mode='bce'):
        if mode == 'bce':
            return F.binary_cross_entropy(input=msk_recons,target=msk)
        
        elif mode == 'mse':
            if self.mwl_apply_gaussian:
                msk,msk_recons = [self.gaussian_smoothing(m) for m in [msk,msk_recons]]
            return ((msk - msk_recons)**2).mean()
        
        else:
            raise NotImplementedError('Only mse or bce are implelented!')

    def forward(self, output, msk1, msk2, occ_masks):
        pyramid_flows = output
        pyramid_mwl_losses = []
        pyramid_occu_masks = occ_masks

        
        for i, (flow, occu_masks) in enumerate(zip(pyramid_flows,pyramid_occu_masks)):
            N, C, H, W, D = flow.size()

            msk1_scaled = F.interpolate(msk1, (H, W, D), mode='nearest')
            # Only needed if we aggregate flow21 and dowing backward computation
            msk2_scaled = F.interpolate(msk2, (H, W, D), mode='nearest')

            flow12 = flow[:, :3]
            flow21 = flow[:, 3:]
            occu1, occu2 = occu_masks
            msk1_recons = flow_warp(msk2_scaled, flow12, mode='nearest')
            msk2_recons = flow_warp(msk1_scaled, flow21, mode='nearest')

            if not self.mwl_use_occ_masks:
                occu1, occu2 = torch.ones_like(occu1), torch.ones_like(occu2)
                                            
            loss_mwl = self.loss_func(msk1_scaled * occu1, msk1_recons * occu1, mode='mse') / occu1.mean()
            if self.w_bk:
                loss_mwl += self.loss_func(msk2_scaled * occu2, msk2_recons * occu2, mode='mse') / occu2.mean()
                loss_mwl /= 2.

            pyramid_mwl_losses.append(loss_mwl)

        pyramid_mwl_losses = [l * w for l, w in zip(pyramid_mwl_losses, self.mwl_w_scales)]
        loss_mwl = sum(pyramid_mwl_losses)
        return loss_mwl

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = [sz//2 for sz in kernel_size]

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)

