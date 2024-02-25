from torch.functional import Tensor
import torchio as tio
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DataAugmentor:
    def __init__(self, args, w_aug, in_shape, out_shape, valid=False, plot=False) -> None:
        self.w_aug = w_aug
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.plot = plot
        self.valid = valid
        self.args = args
        pass
    
    def __call__(self,images, target):
        imgs, img_vox = [im[0] for im in images], [im[1] for im in images]
        #assert(img_vox[0] == img_vox[1]) #! NOTICE THAT VOXEL DIM ARE DIFFERENT 
        #! CHANGED HERE TO NO np.newaxis
        if "DTI" in self.args.model_suffix:
            #sbj_dicts = [{'img': tio.ScalarImage(tensor=im, spacing=vox)} for im,vox in zip(imgs,img_vox)]
            sbj_dicts = [{'img': tio.ScalarImage(tensor=im, affine=np.diag(np.append(vox, 1)))} for im,vox in zip(imgs,img_vox)]
        else:
            sbj_dicts = [{'img': tio.ScalarImage(tensor=im[np.newaxis])} for im,vox in zip(imgs,img_vox)]
        
        if not self.valid and 'masks' in target.keys():
            msks, msk_vox = [im[0] for im in target['masks']], [im[1] for im in target['masks']]
            assert(msk_vox[0] == msk_vox[1] and msk_vox == img_vox)
            [sbj_d.update({'mask':tio.LabelMap(tensor=msk[np.newaxis])}) for sbj_d,msk in zip(sbj_dicts,msks)]
        
        sbjcts = [tio.Subject(sbj_d) for sbj_d in sbj_dicts]
        if self.plot: [sbj.plot() for sbj in sbj_dicts]
        if self.args.resample_value != 0:
            # First resample the images, pad, and then the rest of the transformtions 
            resample = tio.Resample(self.args.resample_value)
            resamples_subjcts = [resample(sbj) for sbj in sbjcts]
            resamples_images = [(aug_sbj['img'].data.squeeze(), vox) for aug_sbj,vox in zip(resamples_subjcts,img_vox)]
            new_img_vox = [(self.args.resample_value, self.args.resample_value, self.args.resample_value), 
                        (self.args.resample_value, self.args.resample_value, self.args.resample_value)]
        
            img1 = resamples_images[0][0] #Shape: (C, W, H, D) 
            img2 = resamples_images[1][0]
        else:
            img1 = imgs[0]
            img2 = imgs[1]
            get_data_width_height_depth(img1, img2)

        ## FIX the padding 
        shape_diff1 = np.abs(np.array(self.out_shape) - np.array(img1.shape[1:]))
        shape_diff2 = np.abs(np.array(self.out_shape) - np.array(img2.shape[1:]))
        


        if self.args.resample_value != 0:
            pad1 = tio.Pad( (int(shape_diff1[0]/2), int(shape_diff1[0]/2), int(shape_diff1[1]/2), int(shape_diff1[1]/2), shape_diff1[2]//2 + 1, shape_diff1[2]//2))
            pad2 = tio.Pad(( int(shape_diff2[0]/2) + 1, int(shape_diff2[0]/2), int(shape_diff2[1]/2) + 1, int(shape_diff2[1]/2), shape_diff2[2]//2 + 1, shape_diff2[2]//2))
            changed_subjcts = [pad1(resamples_subjcts[0]), pad2(resamples_subjcts[1])] #resample + pad 
        else:
            pad1 = tio.Pad( (int(shape_diff1[0]/2), int(shape_diff1[0]/2), int(shape_diff1[1]/2), int(shape_diff1[1]/2), shape_diff1[2]//2 , shape_diff1[2]//2))
            pad2 = tio.Pad(( int(shape_diff2[0]/2) , int(shape_diff2[0]/2), int(shape_diff2[1]/2) , int(shape_diff2[1]/2), shape_diff2[2]//2 , shape_diff2[2]//2))
            changed_subjcts = [pad1(sbjcts[0]), pad2(sbjcts[1])] # pad only 
            new_img_vox = img_vox
        
        
        transforms = get_transforms(self.w_aug, self.valid, self.in_shape, self.out_shape, self.args)
        aug_subjcts = [transforms(sbj) for sbj in changed_subjcts]
        aug_images = [(aug_sbj['img'].data.squeeze(), vox) for aug_sbj,vox in zip(aug_subjcts,new_img_vox)]


        
        if not self.valid and 'masks' in target.keys():
            aug_msks = [(aug_sbj['mask'].data.squeeze(), vox) for aug_sbj,vox in zip(aug_subjcts,msk_vox)]
            target.update({'masks': aug_msks})
        #! PLEASE NOTICE I CHEATED HERE WITH THE SHAPE  (not used now )
        #aug_images[0] = (aug_images[0][0][:, 24:216, :, :], aug_images[0][1])
        
        return aug_images, target

    def get_pair_transformed_images(self,trans_subj: tio.Subject, org_vox_dims, plot=False):
        transformed_imgs = trans_subj.get_images()
        aug_im1 = transformed_imgs[0].data.squeeze()
        aug_im2 = transformed_imgs[1].data.squeeze()
        if plot: trans_subj.plot()
        return aug_im1, aug_im2, org_vox_dims

    def get_transformed_images(self,trans_subj: tio.Subject, org_vox_dims, num_of_images, plot=False):
        transformed_imgs = trans_subj.get_images()
        images = [(transformed_imgs[i].data.squeeze(), org_vox_dims) for i in range(num_of_images)]
        if plot: trans_subj.plot()
        return images

    def pre_validation_set(self,image_tuples, vox, w_aug: bool, plot=False):
        tio_imgs = [tio.Image(tensor=tup[0][np.newaxis], spacing=tup[1]) for tup in image_tuples]
        subj = tio.Subject({f'{idx} image': img for idx, img in enumerate(tio_imgs)})
        if plot: subj.plot()
        transforms = self.get_transforms(w_aug)
        trans_subj = transforms(subj)
        images = self.get_transformed_images(trans_subj, vox, len(image_tuples))
        return images

def no_transform(x):
    return x

def to_float(x):
    return x.type(torch.float)

#def resize_image(img, target_sh):
#    sh_ = img.shape
#    sc = [t/s for t,s in zip(target_sh, sh_)]
#    img_rsz = zoom(img, zoom=sc)
#    return img_rsz

class ResizeImage:
    """
    img - [N,Hi,Wi,Di] 
    shape - [Ho,Wo,Do]
    returns img - [N,Ho,Wo,Do]
    """
    def __init__(self,shape) -> None:
        self.shape = shape
        pass

    def __call__(self, img):
        img = img.unsqueeze(dim=0)
        return F.interpolate(img, size=self.shape, mode='trilinear').squeeze(0)


class RandomCrop:
    def __init__(self, input_shape, target_shape) -> None:
        assert(len(input_shape) == len(target_shape))
        if len(input_shape) == 3:
            h_,w_,d_ = input_shape
            h,w,d = target_shape
            dh, dw, dd = [np.max(l_ - l, 0) for l_,l in zip([h_,w_,d_],[h,w,d])]
            self.st_ = [np.random.randint(0,dl) if dl > 0 else 0 for dl in [dh, dw, dd]]
        else:
            c_,h_,w_,d_ = input_shape
            c,h,w,d = target_shape
            dc, dh, dw, dd = [np.max(l_ - l, 0) for l_,l in zip([c_,h_,w_,d_],[c,h,w,d])]
            self.st_ = [np.random.randint(0,dl) if dl > 0 else 0 for dl in [dc, dh, dw, dd]]
        self.tshape = target_shape
        self.len_input_shape = len(input_shape)
        pass

    def __call__(self, img):
        if len(self.tshape) == 3:
            h,w,d = self.tshape
            hs,ws,ds = self.st_
            return img[:, hs:hs + h, ws:ws + w, ds:ds+d]
        else: #!TAKE NOTICE IF WE SHOULD CUT CHANNELS- no we don't 
            c, h,w,d = self.tshape
            cs, hs,ws,ds = self.st_
            return img[cs:cs + c, hs:hs + h, ws:ws + w, ds:ds+d]


class Crop:
    def __init__(self, input_shape, target_shape) -> None:
        assert(len(input_shape) == len(target_shape))
        if len(input_shape) == 3:
            h_,w_,d_ = input_shape
            h,w,d = target_shape
            dh, dw, dd = [np.max(l_ - l, 0) for l_,l in zip([h_,w_,d_],[h,w,d])]
            self.st_ = [int(dl/2) if dl > 0 else 0 for dl in [dh, dw, dd]]
        else:
            c_,h_,w_,d_ = input_shape
            c,h,w,d = target_shape
            dc, dh, dw, dd = [np.max(l_ - l, 0) for l_,l in zip([c_,h_,w_,d_],[c,h,w,d])]
            self.st_ = [int(dl/2) if dl > 0 else 0 for dl in [dc, dh, dw, dd]]
        self.tshape = target_shape
        self.len_input_shape = len(input_shape)
        pass

    def __call__(self, img):
        if len(self.tshape) == 3:
            h,w,d = self.tshape
            hs,ws,ds = self.st_
            return img[:, hs:hs + h, ws:ws + w, ds:ds+d]
        else: #!TAKE NOTICE IF WE SHOULD CUT CHANNELS- no we don't 
            c, h,w,d = self.tshape
            cs, hs,ws,ds = self.st_
            return img[cs:cs + c, hs:hs + h, ws:ws + w, ds:ds+d]


def get_transforms(w_aug,valid,in_shape,out_shape, args):
    rescale = tio.RescaleIntensity((0, 1))
    tofloat = tio.Lambda(to_float)
    #CHECK IF DTI
    if valid:
        if "DTI" not in args.model_suffix:
            resize_image = tio.Lambda(ResizeImage(out_shape))
            pipe = tio.Compose([tofloat, rescale, resize_image])
        else:
            pipe = tio.Compose([tofloat, rescale])
    else:
        #blur = tio.RandomBlur(std=(3,5), exclude=['img'])
        if "DTI" in args.model_suffix:
            pipe = tio.Compose([tofloat, rescale])
        else:
            random_crop_img = tio.Lambda(RandomCrop(input_shape=in_shape,target_shape=out_shape))
            if w_aug:
                transforms_dict = {tio.RandomAffine(): 0.4, tio.RandomElasticDeformation(): 0.4,
                                tio.RandomFlip(): 0.1, tio.Lambda(no_transform): 0.1}
                pipe = tio.Compose([tofloat, rescale, random_crop_img, tio.OneOf(transforms_dict)])
            else:
                pipe = tio.Compose([tofloat, rescale, random_crop_img])
    return pipe


def get_data_width_height_depth(img1, img2):
    img1_idx_x, img1_idx_y, img1_idx_z = np.nonzero(img1[0])
    img2_idx_x, img2_idx_y, img2_idx_z = np.nonzero(img2[0])
    pass