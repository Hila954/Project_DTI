import numpy as np
from scipy.ndimage import zoom
import scipy.io
import pathlib
from path import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
from abc import ABCMeta
import os
from .loaders import CSVLoader, NiftiLoader, L2RLmsLoader
from .dicom_utils import npz_to_ndarray_and_vox_dim as file_processor
from .dicom_utils import npz_valid_to_ndarrays_flow_vox as vld_file_processor

from .data_augmentor import DataAugmentor

class L2RCTDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, args, valid, root, sp_file, data_seg='trainval', do_aug=True, load_kpts=True, load_masks=False):
        self.root = Path(root)
        self.sp_file = Path(sp_file)
        self.data_seg = data_seg
        self.csv_loader = CSVLoader()
        self.nifti_loader = NiftiLoader()
        self.l2r_kpts_loader = L2RLmsLoader()
        self.load_kpts = load_kpts
        self.load_masks = load_masks
        self.valid = valid
        self.augment = DataAugmentor(args, do_aug, 
                                    in_shape = args.orig_shape, 
                                    out_shape = args.aug_shape if not valid else args.test_shape, 
                                    valid=valid)
        self.samples = self.collect_samples()

    def collect_samples(self):
        samples = []
        scans_dir = self.root / 'training' / 'scans'
        ktps_dir = self.root / 'keypoints'
        scans_list = scans_dir.files('*.gz')
        scans_list.sort()
        pairs = self.load_valid_pairs(csv_file=self.sp_file)

        for idx in range(0,len(scans_list),2):
            file_name = scans_list[idx].parts()[-1]
            csid = int(file_name.split('_')[1])
            if self.data_seg != 'trainval':
                if self.data_seg == 'train' and csid in pairs['fixed']:
                    continue
                if self.data_seg == 'valid' and csid not in pairs['fixed']:
                    continue
            sc_pair = [scans_list[idx], scans_list[idx+1]]
            sample = {'imgs': sc_pair}
            sample['case'] = 'case_{:03d}'.format(csid)
            try:
                assert all([p.isfile() for p in sample['imgs']])
                
                if self.load_masks:
                    mask_dirs = [Path(sc.replace('scans','lungMasks')) for sc in sc_pair]
                    sample['masks'] = mask_dirs
                    assert all([p.isfile() for p in sample['masks']])

                if self.load_kpts:
                    sample['kpts'] = ktps_dir / 'case_{:03d}.csv'.format(csid)
                    assert sample['kpts'].isfile()

            except AssertionError:
                print('Incomplete sample for: {}'.format(sample['imgs'][0]))
                continue

            samples.append(sample)
        return samples
    
    def load_valid_pairs(self,csv_file):
        pairs = self.csv_loader.load(fname=csv_file)
        return {k: [dic[k] for dic in pairs] for k in pairs[0]}
    
    def _load_sample(self, s):
        images  = [self.nifti_loader.load_image(p) for p in s['imgs']]

        target = {'case' : s['case']}
        if 'kpts' in s:
            target['kpts'] = self.l2r_kpts_loader.load(fname=s['kpts'])
        if 'masks' in s:
            masks = [self.nifti_loader.load_image(m) for m in s['masks']]
            target['masks'] = masks

        return images, target, 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images, target = self._load_sample(self.samples[idx])
                
        images, target = self.augment(images, target)
        
        data = {'imgs' : images, 
                'target' : target}
        
        return data 


class CT_4DDataset(Dataset):
    def __init__(self, root: str, w_aug=False,frame_dif=1):
        print(pathlib.Path.cwd())
        root_dir = pathlib.Path(root)
        if not root_dir.exists() or not root_dir.is_dir:
            raise FileExistsError(
                f"{str(root_dir)} doesn't exist or isn't a directory")

        self.root = root_dir
        self.w_augmentations = w_aug
        self.frame_diff = frame_dif

        # Traverse the root directory and count it's size
        self.patient_directories = []
        self.patient_samples = []
        self.collect_samples()

    def __len__(self):
        # We return pairs -> there are len - 1 pairs from len files
        return len(self.patient_samples)

    def __getitem__(self, index):
        img1, vox_dim1 = file_processor(self.patient_samples[index]['img1'])
        img2, vox_dim2 = file_processor(self.patient_samples[index]['img2'])
        sample_name = self.patient_samples[index]['name']
        if img1.shape[2] > 128:
            print('non_mat')
            # todo implement solution

        if self.patient_samples[index]['dim'] == 512:
            img1, vox_dim1 = resize_512_to_256((img1, vox_dim1))
            img2, vox_dim2 = resize_512_to_256((img2, vox_dim2))
        #   img1, img2 = crop_512_imgs_to_256(img1, img2)

        img1, img2 = crop_imgs_to_192_64(img1, img2, 256)
        #img1, img2 = crop_imgs_to_192_64(img1, img2, self.patient_samples[index]['dim'])
        p1, p2 = pre_augmentor(img1, img2, vox_dim1, self.w_augmentations)
        return p1, p2, sample_name

    def collect_samples(self):
        for entry in self.root.iterdir():
            if entry.is_dir():
                self.patient_directories.append(entry)

        self.patient_directories = sorted(self.patient_directories)

        # import zipfile
        for directory in self.patient_directories:
            dir_files = []
            for file in directory.iterdir():
                # print(file)
                # z = zipfile.ZipFile(file)
                # if z.testzip() is not None:
                #     print(file)
                if file.is_file() and file.suffix == '.npz':
                    dir_files.append(file)
            dir_files.sort(key=take_name)
            if len(list(directory.glob('*(256, 256*'))) > 0:
                dim = 256
            else:
                dim = 512
            
            if len(dir_files) < self.frame_diff+1:
                continue
            for idx in range(len(dir_files) - self.frame_diff):
                sample_name = dir_files[idx].name
                sample_name = sample_name[sample_name.index(
                    '_'):sample_name.index('(')]
                name = dir_files[idx].parent.name + sample_name
                self.patient_samples.append(
                    {'name': name, 'img1': dir_files[idx], 'img2': dir_files[idx + self.frame_diff], 'dim': dim})
                # self.patient_samples.append(
                #    {'name': name + '_bk', 'img1': dir_files[idx+1], 'img2': dir_files[idx], 'dim': dim})


class CT_4DValidationset(Dataset):
    def __init__(self, root: str):
        print(pathlib.Path.cwd())
        root_dir = pathlib.Path(root)
        if not root_dir.exists() or not root_dir.is_dir:
            raise FileExistsError(
                f"{str(root_dir)} doesn't exist or isn't a directory")

        self.root = root_dir

        # Traverse the root directory and count it's size
        self.validation_tuples = []
        self.collect_samples()

    def __len__(self):
        return len(self.validation_tuples)

    def __getitem__(self, index):
        p1, p2, flow12 = vld_file_processor(
            self.validation_tuples[index])
        return p1, p2, flow12

    def collect_samples(self):
        for entry in self.root.iterdir():
            dir_files = []
            if entry.is_file():
                dir_files.append(entry)
            dir_files.sort()
            for idx in range(len(dir_files)):
                self.validation_tuples.append(dir_files[idx])

class DTI_Dataset_example(Dataset, metaclass=ABCMeta):
    def __init__(self, args, valid, root, sp_file, data_seg='trainval', do_aug=True, load_kpts=False, load_masks=False):
        print(pathlib.Path.cwd())
        self.data_seg = data_seg
        self.sp_file = Path(sp_file)
        self.csv_loader = CSVLoader()
        self.augment = DataAugmentor(args, do_aug, 
                                    in_shape = args.orig_shape, 
                                    out_shape = args.aug_shape if not valid else args.test_shape, 
                                    valid=valid)
        self.root = pathlib.Path(root)
        self.data_Hyrax_name = 'padded_DTI_Hyrax_data.mat'
        self.data_Dog_name = 'shifted_padded_DTI_Dog_data.mat'

        #######################################################################! HERE 
        self.samples = self.collect_samples()

    def collect_samples(self):
        samples = []
        scans_dir = Path(self.root) 
        scans_list = scans_dir.files('*.mat')
        scans_list.sort()
        pairs = self.load_valid_pairs(csv_file=self.sp_file)

        for idx in range(0,len(scans_list),2):
            file_name = scans_list[idx].parts()[-1]
            csid = int(file_name.split('_')[1])
            # if self.data_seg != 'trainval':
                # if self.data_seg == 'train' and csid in pairs['fixed']:
                #     continue
                # if self.data_seg == 'valid' and csid not in pairs['fixed']:
                #     continue
            sc_pair = [scans_list[idx], scans_list[idx+1]]
            sample = {'imgs': sc_pair}
            sample['case'] = 'case_{:03d}'.format(csid)
            try:
                assert all([p.isfile() for p in sample['imgs']])

            except AssertionError:
                print('Incomplete sample for: {}'.format(sample['imgs'][0]))
                continue

            samples.append(sample)
        return samples
    
    def load_valid_pairs(self,csv_file):
        pairs = self.csv_loader.load(fname=csv_file)
        return {k: [dic[k] for dic in pairs] for k in pairs[0]}
    
    def _load_sample(self, s):
        #! CHANGE TO GENERAL ACCESS 
        img2_mat_env = scipy.io.loadmat(s['imgs'][0])
        img2 = img2_mat_env['DT_6C']
        img2[np.isnan(img2)] = 0
        #! padding like CT
        img2 = np.pad(img2, [(0, 0), (32, 32), (48, 48), (0, 66)], mode='constant', constant_values=0)
        #img1 = img1[:2,:,:,:]
        img1_mat_env = scipy.io.loadmat(s['imgs'][1])
        img1 = img1_mat_env['shifted_padded_DT_6C']
        img1[np.isnan(img1)] = 0
        img1 = np.pad(img1, [(0, 0), (32, 32), (48, 48), (0, 64)], mode='constant', constant_values=0)
        #img2 = img2[:2,:,:,:]
        images  = [(img1,(0.4, 0.4, 0.4)), (img2,(0.4, 0.4, 0.4))]
        target = {'case' : s['case']}
        
        return images, target, 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images, target = self._load_sample(self.samples[idx])
                
        images, target = self.augment(images, target)
        
        data = {'imgs' : images, 
                'target' : target}
        
        return data 



class CT_4D_Variance_Valid_set(CT_4DDataset):
    def __init__(self, root: str, w_aug=False, set_length=10, num_of_sets=15):
        self.set_length = set_length
        self.num_of_sets = num_of_sets
        super().__init__(root, w_aug=w_aug)
        # Traverse the root directory and count it's size
        self.patient_directories = []
        self.patient_samples = []
        self.collect_samples()

    def __getitem__(self, index):
        img1, vox_dim1 = file_processor(self.patient_samples[index]['img1'])
        img2, vox_dim2 = file_processor(self.patient_samples[index]['img2'])
        sample_name = self.patient_samples[index]['name']
        if img1.shape[2] > 128:
            print('non_mat')
            # todo implement solution

        if self.patient_samples[index]['dim'] == 512:
            img1, vox_dim1 = resize_512_to_256((img1, vox_dim1))
            img2, vox_dim2 = resize_512_to_256((img2, vox_dim2))

        p1, p2 = pre_augmentor(img1, img2, vox_dim1, False)
        return p1, p2, sample_name
        # idx = [i + 1 for i in range(self.set_length)]
        # image_tuples = []
        # for i in idx:
        #     image_tuples.append(file_processor(self.patient_sets[index][f'img{i}']))

        # sample_name = self.patient_sets[index]['name']
        # if image_tuples[0][0].shape[2] > 128:
        #     print('non_mat')
        #     # todo implement solution

        # if self.patient_sets[index]['dim'] == 512:
        #     for idx in range(len(image_tuples)):
        #         image_tuples[idx] = resize_512_to_256(image_tuples[idx])
        # image_tuples = pre_validation_set(image_tuples,vox=image_tuples[0][1], w_aug=self.w_augmentations)
        # return image_tuples

    def collect_samples(self):
        for entry in self.root.iterdir():
            if entry.is_dir():
                self.patient_directories.append(entry)

        self.patient_directories = sorted(self.patient_directories)

        for directory in self.patient_directories:
            if len(self.patient_samples) >= self.num_of_sets * (self.set_length-1):
                break
            dir_files = []
            for file in directory.iterdir():
                if file.is_file() and file.suffix == '.npz':
                    dir_files.append(file)
            dir_files.sort(key=take_name)

            if len(list(directory.glob('*(256, 256*'))) > 0:
                dim = 256
            else:
                dim = 512
            if len(dir_files) < self.set_length:
                continue
            for idx in range(self.set_length - 1):
                sample_name = dir_files[idx].name
                sample_name = sample_name[sample_name.index(
                    '_'):sample_name.index('(')]
                name = dir_files[idx].parent.name + sample_name
                self.patient_samples.append(
                    {'name': name, 'img1': dir_files[idx], 'img2': dir_files[idx + 1], 'dim': dim})
            # adding last sample as apair of first image and last image of set
            # self.patient_samples.append(
            #    {'name':dir_files[idx].parent.name, 'img1':dir_files[0], 'img2':dir_files[self.set_length-1], 'dim':dim})

            # if len(dir_files) < self.set_length:
            #     continue
            # idx = [i for i in range(self.set_length)]
            # name = dir_files[idx[0]].parent.name
            # set_dict = {f'img{i + 1}': dir_files[i] for i in idx}
            # set_dict['name'] = name
            # set_dict['dim'] = dim

            # self.patient_sets.append(set_dict)


def pad_img_to_128(img):
    pad = np.zeros((img.shape[0], img.shape[1], 128))
    pad[:img.shape[0], :img.shape[1], :img.shape[2]] = img
    return pad


def crop_512_imgs_to_256(image1, image2):
    # returns a random quarter of an image
    i = np.random.randint(2)
    j = np.random.randint(2)
    return (
        image1[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :],
        image2[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :])


def random_crop(img, target_shape, mask=None):
    assert(len(img.shape) == 3)
    if mask is not None:
        assert(img.shape == mask.shape)

    h_,w_,d_ = img.shape
    h,w,d =  target_shape
    
    dh = np.random.randint(0, np.max(0, h_ - h))
    dw = np.random.randint(0, np.max(0, w_ - w))
    dd = np.random.randint(0, np.max(0, d_ - d))

    img = img[dh:dh + h, dw:dw + w, dd:dd+d]
    
    if mask is not None:
        mask = mask[dh:dh + h, dw:dw + w, dd:dd+d]

    return img, mask


def resize_512_to_256(img_tup):
    img = img_tup[0]
    z = img.shape[2]
    vox_dim = img_tup[1]
    img = zoom(img, zoom=[0.5, 0.5, 1],order=1)
    vox_dim[0] = vox_dim[0] * 2
    vox_dim[1] = vox_dim[1] * 2

    return img, vox_dim


def get_dataset(args,root="./raw", w_aug=False, data_type='train',frame_dif=1,sp_filepth="",valid=False):
    if data_type =='l2r_train':
        return L2RCTDataset(args, valid, root=root,sp_file=args.sp_filepth_train, data_seg='train',do_aug=w_aug,load_kpts=False,load_masks=True)
    if data_type =='l2r_valid':
        return L2RCTDataset(args, valid, root=root,sp_file=args.sp_filepth_valid, data_seg='valid',do_aug=w_aug,load_kpts=True,load_masks=True)
    if data_type =='l2r_test':
        return L2RCTDataset(args, valid, root=root,sp_file=args.sp_filepth_valid, data_seg='valid',do_aug=w_aug,load_kpts=False,load_masks=False)


    if data_type == 'train':
        return CT_4DDataset(root=root, w_aug=w_aug, frame_dif=frame_dif)
    if data_type == 'synthetic':
        # return CT_4DDataset(root=root, w_aug=w_aug)
        return CT_4DValidationset(root)
    if data_type == 'variance_valid':
        return CT_4D_Variance_Valid_set(root=root, w_aug=w_aug)
    
    if data_type =='DTI_Example_train':
        return DTI_Dataset_example(args, valid, root=root,sp_file=args.sp_filepth_train, data_seg='train',do_aug=w_aug,load_kpts=False,load_masks=True)
    if data_type =='DTI_Example_valid':
        return DTI_Dataset_example(args, valid, root=root,sp_file=args.sp_filepth_valid, data_seg='valid',do_aug=w_aug,load_kpts=True,load_masks=True)
    if data_type =='DTI_Example_test':
        return DTI_Dataset_example(args, valid, root=root,sp_file=args.sp_filepth_valid, data_seg='valid',do_aug=w_aug,load_kpts=False,load_masks=False)



def take_name(file_path):
    name = file_path.name
    name = name[name.index('_') + 1:(name.index('(') - 1)]
    return int(name)

