import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from scipy.ndimage.interpolation import map_coordinates, zoom


def mesh_grid(B, H, W, D):
    # batches not implented
    x = torch.arange(H)
    y = torch.arange(W)
    z = torch.arange(D)
    mesh = torch.stack(torch.meshgrid(x, y, z)[::-1], 0)
    mesh = mesh.unsqueeze(0)
    return mesh.repeat([B,1,1,1,1])


def norm_grid(v_grid):
    _, _, H, W, D = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (H - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 2, :, :] = 2.0 * v_grid[:, 2, :, :] / (D - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 4, 1)

def norm_lms(lms,flow_shape):
    _, _, H, W, D = flow_shape

    # scale grid to [-1,1]
    lms_norm = torch.zeros_like(lms)
    lms_norm[:, 0, :, :] = 2.0 * lms[:, 0, :, :] / (D - 1) - 1.0
    lms_norm[:, 1, :, :] = 2.0 * lms[:, 1, :, :] / (H - 1) - 1.0
    lms_norm[:, 2, :, :] = 2.0 * lms[:, 2, :, :] / (W - 1) - 1.0
    return lms_norm.permute(0, 2, 3, 4, 1)


def flow_warp(img2, flow12, pad='border', mode='bilinear'):
    B, _, H, W, D = flow12.size()
    flow12 = torch.flip(flow12, [1])
    base_grid = mesh_grid(B, H, W, D).type_as(img2)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    im1_recons = nn.functional.grid_sample(
        img2, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    return im1_recons


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()

def resize_flow_tensor(flow,shape):
    """
    flow - [C,H,W,D] 
    shape - [H,W,D]
    returns flow - [1,C,H,W,D]
    """
    sc = [t/s for t,s in zip(shape, flow.shape[1:])]
    flow = flow.unsqueeze(dim=0)
    flow = F.interpolate(flow, size=shape, mode='trilinear')
    return flow * torch.tensor(sc,device=flow.device)[None,:,None,None,None]

def evaluate_flow(pred_flow, kpts, masks, spacing):
    lms_fixed = np.array(kpts['lms_fixed'])
    lms_moving = np.array(kpts['lms_moving'])
    spacing = np.array(spacing.cpu())
    img = np.array(masks[0][0]).squeeze(0)

    # resize flow
    pred_flow = resize_flow_tensor(pred_flow, shape=img.shape)
    
    # calc log(determinant) of jaciboan
    pred_flow = pred_flow.cpu().numpy()
    jac_det = (jacobian_determinant(pred_flow) + 3).clip(0.000000001, 1000000000)
    log_jac_det = np.log(jac_det)
    
    # calc tre
    pred_flow = pred_flow.squeeze()
    # fix dims
    disp_half = np.array([zoom(pred_flow_, 0.5, order=2).astype('float16') for pred_flow_ in pred_flow])
    pred_flow = np.array([zoom(disp_half[i].astype('float32'), 2, order=2) for i in range(3)])

    #disp_x = zoom(pred_flow[0], 0.5, order=2).astype('float16')
    #disp_y = zoom(pred_flow[1], 0.5, order=2).astype('float16')
    #disp_z = zoom(pred_flow[2], 0.5, order=2).astype('float16')
    #disp = np.array((disp_x, disp_y, disp_z))

    lms_fixed_disp_x = map_coordinates(pred_flow[0], lms_fixed.transpose())
    lms_fixed_disp_y = map_coordinates(pred_flow[1], lms_fixed.transpose())
    lms_fixed_disp_z = map_coordinates(pred_flow[2], lms_fixed.transpose())
    lms_fixed_disp = np.array((lms_fixed_disp_x, lms_fixed_disp_y, lms_fixed_disp_z)).transpose()
    lms_fixed_warped = lms_fixed + lms_fixed_disp
    tre = compute_tre(lms_fixed_warped, lms_moving, spacing)

    return tre.mean(), np.ma.MaskedArray(log_jac_det, 1-img[2:-2, 2:-2, 2:-2]).std()

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet

def compute_tre(x, y, spacing):
    return np.linalg.norm((x - y) * spacing, axis=1)

