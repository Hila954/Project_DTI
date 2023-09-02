import numpy as np
import matplotlib.pyplot as plt
import torch
import flow_vis
from utils.flow_utils import flow_warp, resize_flow_tensor
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure
import torch.nn.functional as F
import matplotlib.cm as cm
from scipy.ndimage.interpolation import map_coordinates


def plot_3d(image, mask, kpts, title, show_mesh=False): 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    if show_mesh:
        # draw shape using marching cubes
        threshold = image[np.where(mask)].mean()
        verts, faces, _, _ = measure.marching_cubes(image,threshold)
        mesh = Poly3DCollection(verts[faces], alpha=0.1)
        face_color = [0.18, 0.31, 0.31]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

    # add kpts
    n = kpts[0].shape[0]
    #i = np.random.permutation(n)
    colors = cm.rainbow(np.linspace(0, 1, n))
    markers = ['^','o']
    lables = ['gt kpts','pred kpts']
    for kpts_,m,lab in zip(kpts,markers,lables):
        xp, yp, zp = kpts_.T
        ax.scatter(xp, yp, zp, marker=m, c=colors,label=lab)
    
    # view adjusments
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)

    return fig

def plot_imgs_and_lms(imgs,masks,kpts,flow12):
    lms_fixed = np.array(kpts['lms_fixed'])
    lms_moving = np.array(kpts['lms_moving'])

    masks = [m[0].cpu().squeeze().numpy() for m in masks]
    imgs = [img.cpu() for img in imgs]

    imgs = [F.interpolate(img, size=m.shape, mode='trilinear') for img,m in zip(imgs,masks)]
    flow12 = resize_flow_tensor(flow12, shape=masks[0].shape).cpu().numpy()

    flow12 = flow12.squeeze()
    lms_fixed_disp_x = map_coordinates(flow12[0], lms_fixed.transpose())
    lms_fixed_disp_y = map_coordinates(flow12[1], lms_fixed.transpose())
    lms_fixed_disp_z = map_coordinates(flow12[2], lms_fixed.transpose())
    lms_fixed_disp = np.array((lms_fixed_disp_x, lms_fixed_disp_y, lms_fixed_disp_z)).transpose()
    lms_fixed_warped = lms_fixed + lms_fixed_disp

    titles = ['input','warped']
    figs = [plot_3d(img.squeeze().numpy(), mask, kpts, tit_) 
        for img,mask,kpts, tit_ in  zip(imgs,masks,[[lms_moving,lms_fixed],[lms_moving,lms_fixed_warped]],titles)]

    return figs

def plot_image(
        data,
        axes=None,
        output_path=None,
        show=True,
):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    while len(data.shape) > 3:
        data = data.squeeze(0)
    indices = np.array(data.shape) // 2
    i, j, k = indices
    k=32
    slice_x = rotate(data[i, :, :])
    slice_y = rotate(data[:, j, :])
    slice_z = rotate(data[:, :, k])

    kwargs = {}
    kwargs['cmap'] = 'YlGnBu'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in data.shape]
    f0 = axes[0].imshow(slice_x, extent=y_extent + z_extent, **kwargs)
    f1 = axes[1].imshow(slice_y, extent=x_extent + z_extent, **kwargs)
    f2 = axes[2].imshow(slice_z, extent=x_extent + y_extent, **kwargs)
    plt.colorbar(f0, ax=axes[0])
    plt.colorbar(f1, ax=axes[1])
    plt.colorbar(f2, ax=axes[2])
    plt.tight_layout()
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def plot_images(
        img1, img2, img3,
        axes=None,
        output_path=None,
        show=True,):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img2.shape) > 3:
        img2 = img2.squeeze(0)
    while len(img3.shape) > 3:
        img3 = img3.squeeze(0)

    indices = np.array(img1.shape) // 2
    i, j, k = indices
    k=32
    slice_x_1 = rotate(img1[i, :, :])
    slice_y_1 = rotate(img1[:, j, :])
    slice_z_1 = rotate(img1[:, :, k])
    slice_x_2 = rotate(img2[i, :, :])
    slice_y_2 = rotate(img2[:, j, :])
    slice_z_2 = rotate(img2[:, :, k])
    slice_x_3 = rotate(img3[i, :, :])
    slice_y_3 = rotate(img3[:, j, :])
    slice_z_3 = rotate(img3[:, :, k])
    kwargs = {}
    kwargs['cmap'] = 'gray'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in img1.shape]
    axes[0][0].imshow(slice_x_1, extent=y_extent + z_extent, **kwargs)
    axes[0][1].imshow(slice_y_1, extent=x_extent + z_extent, **kwargs)
    axes[0][2].imshow(slice_z_1, extent=x_extent + y_extent, **kwargs)
    axes[1][0].imshow(slice_x_2, extent=y_extent + z_extent, **kwargs)
    axes[1][1].imshow(slice_y_2, extent=x_extent + z_extent, **kwargs)
    axes[1][2].imshow(slice_z_2, extent=x_extent + y_extent, **kwargs)
    axes[2][0].imshow(slice_x_3, extent=y_extent + z_extent, **kwargs)
    axes[2][1].imshow(slice_y_3, extent=x_extent + z_extent, **kwargs)
    axes[2][2].imshow(slice_z_3, extent=x_extent + y_extent, **kwargs)
    plt.tight_layout()
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def rotate(image):
    return np.rot90(image)


def plot_flow(flow,
              axes=None,
              output_path=None,
              show=True, ):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    while len(flow.shape) > 4:
        flow = flow.squeeze(0)
    indices = np.array(flow.shape[1:]) // 2
    i, j, k = indices
    k=32

    slice_x_flow = (flow[1:3, i, :, :])
    slice_x_flow_col = rotate(flow_vis.flow_to_color(
        slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
    slice_y_flow_col = rotate(flow_vis.flow_to_color(
        slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_z_flow = (flow[0:2, :, :, k])
    slice_z_flow_col = rotate(flow_vis.flow_to_color(
        slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    # xy_grid = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    # xz_grid = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[2]))
    kwargs = {}
    # kwargs['cmap'] = 'gray'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in flow.shape[1:]]
    axes[0].imshow(slice_x_flow_col, extent=y_extent + z_extent, **kwargs)
    axes[1].imshow(slice_y_flow_col, extent=x_extent + z_extent, **kwargs)
    axes[2].imshow(slice_z_flow_col, extent=x_extent + y_extent, **kwargs)
    plt.tight_layout()

    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    # return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
    return fig


def plot_training_fig(img1, img2, flow,
                      axes=None,
                      output_path=None,
                      show=True, ):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img2.shape) > 3:
        img2 = img2.squeeze(0)
    while len(flow.shape) > 4:
        flow = flow.squeeze(0)
    indices = np.array(flow.shape[1:]) // 2
    i, j, k = indices
    k=32

    slice_x_1 = rotate(img1[i, :, :])
    slice_y_1 = rotate(img1[:, j, :])
    slice_z_1 = rotate(img1[:, :, k])
    slice_x_2 = rotate(img2[i, :, :])
    slice_y_2 = rotate(img2[:, j, :])
    slice_z_2 = rotate(img2[:, :, k])
    slice_x_flow = (flow[1:3, i, :, :])
    slice_x_flow_col = rotate(flow_vis.flow_to_color(
        slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
    slice_y_flow_col = rotate(flow_vis.flow_to_color(
        slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_z_flow = (flow[0:2, :, :, k])
    slice_z_flow_col = rotate(flow_vis.flow_to_color(
        slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    kwargs = {}
    kwargs['cmap'] = 'gray'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in flow.shape[1:]]
    axes[0][0].imshow(slice_x_1, extent=y_extent + z_extent, **kwargs)
    axes[0][1].imshow(slice_y_1, extent=x_extent + z_extent, **kwargs)
    axes[0][2].imshow(slice_z_1, extent=x_extent + y_extent, **kwargs)
    axes[1][0].imshow(slice_x_2, extent=y_extent + z_extent, **kwargs)
    axes[1][1].imshow(slice_y_2, extent=x_extent + z_extent, **kwargs)
    axes[1][2].imshow(slice_z_2, extent=x_extent + y_extent, **kwargs)
    axes[2][0].imshow(slice_x_flow_col, extent=y_extent + z_extent)
    axes[2][1].imshow(slice_y_flow_col, extent=x_extent + z_extent)
    axes[2][2].imshow(slice_z_flow_col, extent=x_extent + y_extent)
    plt.tight_layout()

    if output_path is not None and fig is not None:
        fig.savefig(output_path, format='jpg')
    if show:
        plt.show()
    # return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
    return fig

def plot_validation_fig(img1, img2, flow_gt, flow,
                        axes=None,
                        output_path=None,
                        show=True, ):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(4, 3, figsize=(10, 10))

    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img2.shape) > 3:
        img2 = img2.squeeze(0)
    while len(flow_gt.shape) > 4:
        flow_gt = flow_gt.squeeze(0)
    while len(flow.shape) > 4:
        flow = flow.squeeze(0)
    indices = np.array(flow.shape[1:]) // 2
    i, j, k = indices
    k=32

    slice_x_1 = rotate(img1[i, :, :])
    slice_y_1 = rotate(img1[:, j, :])
    slice_z_1 = rotate(img1[:, :, k])
    slice_x_2 = rotate(img2[i, :, :])
    slice_y_2 = rotate(img2[:, j, :])
    slice_z_2 = rotate(img2[:, :, k])
    slice_x_flow = (flow[1:3, i, :, :])
    slice_x_flow_col = rotate(flow_vis.flow_to_color(
        slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
    slice_y_flow_col = rotate(flow_vis.flow_to_color(
        slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_z_flow = (flow[0:2, :, :, k])
    slice_z_flow_col = rotate(flow_vis.flow_to_color(
        slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_x_flow_gt = (flow_gt[1:3, i, :, :])
    slice_x_flow_col_gt = rotate(flow_vis.flow_to_color(
        slice_x_flow_gt.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_y_flow_gt = (torch.stack((flow_gt[0, :, j, :], flow_gt[2, :, j, :])))
    slice_y_flow_col_gt = rotate(flow_vis.flow_to_color(
        slice_y_flow_gt.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_z_flow_gt = (flow_gt[0:2, :, :, k])
    slice_z_flow_col_gt = rotate(flow_vis.flow_to_color(
        slice_z_flow_gt.permute([1, 2, 0]).numpy(), convert_to_bgr=False))

    kwargs = {}
    kwargs['cmap'] = 'gray'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in flow.shape[1:]]
    axes[0][0].imshow(slice_x_1, extent=y_extent + z_extent, **kwargs)
    axes[0][1].imshow(slice_y_1, extent=x_extent + z_extent, **kwargs)
    axes[0][2].imshow(slice_z_1, extent=x_extent + y_extent, **kwargs)
    axes[1][0].imshow(slice_x_2, extent=y_extent + z_extent, **kwargs)
    axes[1][1].imshow(slice_y_2, extent=x_extent + z_extent, **kwargs)
    axes[1][2].imshow(slice_z_2, extent=x_extent + y_extent, **kwargs)
    axes[2][0].imshow(slice_x_flow_col, extent=y_extent + z_extent)
    axes[2][1].imshow(slice_y_flow_col, extent=x_extent + z_extent)
    axes[2][2].imshow(slice_z_flow_col, extent=x_extent + y_extent)
    axes[3][0].imshow(slice_x_flow_col_gt, extent=y_extent + z_extent)
    axes[3][1].imshow(slice_y_flow_col_gt, extent=x_extent + z_extent)
    axes[3][2].imshow(slice_z_flow_col_gt, extent=x_extent + y_extent)
    plt.tight_layout()

    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    # return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
    return fig


def plot_warped_img(img1, img1_recons, axes=None, output_path=None, show=False):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img1_recons.shape) > 3:
        img1_recons = img1_recons.squeeze(0)
    indices = np.array(img1.shape) // 2
    i, j, k = indices
    k=32
    slice_x_r = rotate(img1[i, :, :])
    slice_x_g = rotate(img1_recons[i, :, :])
    slice_x_b = (slice_x_r+slice_x_g)/2
    slice_x = np.dstack((slice_x_r, slice_x_g, slice_x_b))

    slice_y_r = rotate(img1[:, j, :])
    slice_y_g = rotate(img1_recons[:, j, :])
    slice_y_b = (slice_y_r+slice_y_g)/2
    slice_y = np.dstack((slice_y_r, slice_y_g, slice_y_b))
    
    slice_z_r = rotate(img1[:, :, k])
    slice_z_g = rotate(img1_recons[:, :, k])
    slice_z_b = (slice_z_r+slice_z_g)/2
    slice_z = np.dstack((slice_z_r, slice_z_g, slice_z_b))

    kwargs = {}
    x_extent, y_extent, z_extent = [(0, b - 1) for b in img1.shape]
    f0 = axes[0].imshow(slice_x, extent=y_extent + z_extent, **kwargs)
    f1 = axes[1].imshow(slice_y, extent=x_extent + z_extent, **kwargs)
    f2 = axes[2].imshow(slice_z, extent=x_extent + y_extent, **kwargs)

    img = np.concatenate([slice_x, slice_y, slice_z],axis=1)
    plt.tight_layout()
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig

def disp_warped_img(img1, img1_recons):
    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img1_recons.shape) > 3:
        img1_recons = img1_recons.squeeze(0)
    
    indices = np.array(img1.shape) // 2
    i, j, k = indices
    k=32
    slice_x_r = rotate(img1[i, :, :])
    slice_x_g = rotate(img1_recons[i, :, :])
    slice_x_b = (slice_x_r+slice_x_g)/2
    slice_x = np.dstack((slice_x_r, slice_x_g, slice_x_b))
    
    slice_y_r = rotate(img1[:, j, :])
    slice_y_g = rotate(img1_recons[:, j, :])
    slice_y_b = (slice_y_r+slice_y_g)/2
    slice_y = np.dstack((slice_y_r, slice_y_g, slice_y_b))
    
    slice_z_r = rotate(img1[:, :, k])
    slice_z_g = rotate(img1_recons[:, :, k])
    slice_z_b = (slice_z_r+slice_z_g)/2
    slice_z = np.dstack((slice_z_r, slice_z_g, slice_z_b))
    #! to fix different slices dimensions 
    slice_z_t = slice_z[:slice_x.shape[0],:,:]
    return np.concatenate([slice_x, slice_y, slice_z_t],axis=1)[None,::]

def disp_training_fig(img1, img2, flow):
    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img2.shape) > 3:
        img2 = img2.squeeze(0)
    while len(flow.shape) > 4:
        flow = flow.squeeze(0)
    indices = np.array(flow.shape[1:]) // 2
    i, j, k = indices
    k=32

    slice_x_1 = rotate(img1[i, :, :])
    slice_y_1 = rotate(img1[:, j, :])
    slice_z_1 = rotate(img1[:, :, k])
    slice_x_2 = rotate(img2[i, :, :])
    slice_y_2 = rotate(img2[:, j, :])
    slice_z_2 = rotate(img2[:, :, k])
    
    slice_x_flow = (flow[1:3, i, :, :])
    slice_x_flow_col = rotate(flow_vis.flow_to_color(
        slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
    slice_y_flow_col = rotate(flow_vis.flow_to_color(
        slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_z_flow = (flow[0:2, :, :, k])
    slice_z_flow_col = rotate(flow_vis.flow_to_color(
        slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    kwargs = {}
    kwargs['cmap'] = 'gray'

    #! fixing dimensions
    slice_z_1 = slice_z_1[:slice_x_1.shape[0], :]
    slice_z_2 = slice_z_2[:slice_x_2.shape[0], :]
    slice_z_flow_col = slice_z_flow_col[:slice_y_flow_col.shape[0], :, :]

    slice_dx_middle = flow[0, i, :, :].numpy()
    slice_dy_middle = flow[1, i, :, :].numpy()
    slice_dz_middle = flow[2, i, :, :].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    im = axes[0].imshow(slice_dx_middle)
    axes[1].imshow(slice_dy_middle)
    axes[2].imshow(slice_dz_middle)
    plt.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal")

    slices_1 = [np.tile(sl_,(3,1,1)) for sl_ in [slice_x_1,slice_y_1,slice_z_1]]
    slices_2 = [np.tile(sl_,(3,1,1)) for sl_ in [slice_x_2,slice_y_2,slice_z_2]]
    flows12 = [np.transpose(fl_,(2,0,1)) for fl_ in [slice_x_flow_col,slice_y_flow_col,slice_z_flow_col]]
 
    slice_imgs = [np.concatenate([s1,s2,f12],axis=1) for s1,s2,f12 in zip(slices_1,slices_2,flows12)]
    return np.concatenate(slice_imgs,axis=2)[None,::], fig

