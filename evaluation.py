from evalutils.exceptions import ValidationError
from evalutils.io import CSVLoader, FileLoader, ImageLoader
import json
import nibabel as nib
import numpy as np
import os.path
from pathlib import Path
from pandas import DataFrame, MultiIndex
import scipy.ndimage
from scipy.ndimage.interpolation import map_coordinates, zoom
#from surface_distance import *

##### paths #####

DEFAULT_INPUT_PATH = Path("/home/gallif/projects/optical_flow/_4DCTCostUnrolling/submission")
DEFAULT_GROUND_TRUTH_PATH = Path("/home/gallif/datasets/L2R2021")
DEFAULT_EVALUATION_OUTPUT_FILE_PATH = Path("/metrics.json")

##### metrics #####

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

##### file loader #####

class NiftiLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return nib.load(str(fname))

    @staticmethod
    def hash_image(image):
        return hash(image.get_fdata().tostring())
    
class NumpyLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return np.load(str(fname))['arr_0']

    @staticmethod
    def hash_image(image):
        return hash(image.tostring())
    
class CURIOUSLmsLoader(FileLoader):
    def load(self, fname):
        lms_fixed = []
        lms_moving = []
        f = open(fname, 'r')
        for line in f.readlines()[5:]:
            lms = [float(lm) for lm in line.split(' ')[1:-1]]
            lms_fixed.append(lms[:3])
            lms_moving.append(lms[3:])
        return {'lms_fixed': lms_fixed, 'lms_moving': lms_moving}
    
class L2RLmsLoader(FileLoader):
    def load(self, fname):
        lms_fixed = []
        lms_moving = []
        f = open(fname, 'r')
        for line in f.readlines():
            lms = [float(lm) for lm in line.split(',')]
            lms_fixed.append(lms[:3])
            lms_moving.append(lms[3:])
        return {'lms_fixed': lms_fixed, 'lms_moving': lms_moving}
    
##### validation errors #####
def raise_missing_file_error(fname):
    message = (
        f"The displacement field {fname} is missing. "
        f"Please provide all required displacement fields."
    )
    raise ValidationError(message)
    
def raise_dtype_error(fname, dtype):
    message = (
        f"The displacement field {fname} has a wrong dtype ('{dtype}'). "
        f"All displacement fields should have dtype 'float16'."
    )
    raise ValidationError(message)
    
def raise_shape_error(fname, shape, expected_shape):
    message = (
        f"The displacement field {fname} has a wrong shape ('{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}'). "
        f"The expected shape of displacement fields for this task is {expected_shape[0]}x{expected_shape[1]}x{expected_shape[2]}x{expected_shape[3]}."
    )
    raise ValidationError(message)
        
##### eval val #####

class EvalVal():
    def __init__(self):
        self.ground_truth_path = DEFAULT_GROUND_TRUTH_PATH
        self.predictions_path = DEFAULT_INPUT_PATH
        self.output_file = DEFAULT_EVALUATION_OUTPUT_FILE_PATH
        
        self.csv_loader = CSVLoader()
        self.nifti_loader = NiftiLoader()
        self.numpy_loader = NumpyLoader()
        self.curious_lms_loader = CURIOUSLmsLoader()
        self.l2r_lms_loader = L2RLmsLoader()
        
        self.pairs_task_01 = DataFrame()
        self.imgs_task_01 = DataFrame()
        self.lms_task_01 = DataFrame()
        self.disp_fields_task_01 = DataFrame()
        self.cases_task_01 = DataFrame()
        
        self.pairs_task_02 = DataFrame()
        self.imgs_task_02 = DataFrame()
        self.lms_task_02 = DataFrame()
        self.disp_fields_task_02 = DataFrame()
        self.cases_task_02 = DataFrame()
        
        self.pairs_task_03 = DataFrame()
        self.segs_task_03 = DataFrame()
        self.disp_fields_task_03 = DataFrame()
        self.cases_task_03 = DataFrame()
        
        self.pairs_task_04 = DataFrame()
        self.segs_task_04 = DataFrame()
        self.disp_fields_task_04 = DataFrame()
        self.cases_task_04 = DataFrame()
    
    def evaluate(self):
#        self.load_task_01()
#        self.merge_ground_truth_and_predictions_task_01()
#        self.score_task_01()
        
        self.load_task_02()
        self.merge_ground_truth_and_predictions_task_02()
        self.score_task_02()
        
 #       self.load_task_03()
 #       self.merge_ground_truth_and_predictions_task_03()
 #       self.score_task_03()
        
#        self.load_task_04()
#        self.merge_ground_truth_and_predictions_task_04()
#        self.score_task_04()
        
        self.save()
        
    def load_task_01(self):
        self.pairs_task_01 = self.load_pairs(DEFAULT_GROUND_TRUTH_PATH / 'task_01' / 'pairs_val.csv')
        self.imgs_task_01 = self.load_imgs_task_01()
        self.lms_task_01 = self.load_lms_task_01()
        self.disp_fields_task_01 = self.load_disp_fields(self.pairs_task_01, DEFAULT_INPUT_PATH / 'task_01', np.array([3, 128, 128, 144]))
        
    def load_task_02(self):
        self.pairs_task_02 = self.load_pairs(DEFAULT_GROUND_TRUTH_PATH / 'task_02' / 'pairs_val.csv')
        self.imgs_task_02 = self.load_imgs_task_02()
        self.lms_task_02 = self.load_lms_task_02()
        self.disp_fields_task_02 = self.load_disp_fields(self.pairs_task_02, DEFAULT_INPUT_PATH / 'task_02', np.array([3, 96, 96, 104]))
        
    def load_task_03(self):
        self.pairs_task_03 = self.load_pairs(DEFAULT_GROUND_TRUTH_PATH / 'task_03' / 'pairs_val.csv')
        self.segs_task_03 = self.load_segs_task_03()
        self.disp_fields_task_03 = self.load_disp_fields(self.pairs_task_03, DEFAULT_INPUT_PATH / 'task_03', np.array([3, 96, 80, 128]))
        
    def load_task_04(self):
        self.pairs_task_04 = self.load_pairs(DEFAULT_GROUND_TRUTH_PATH / 'task_04' / 'pairs_val.csv')
        self.segs_task_04 = self.load_segs_task_04()
        self.disp_fields_task_04 = self.load_disp_fields(self.pairs_task_04, DEFAULT_INPUT_PATH / 'task_04', np.array([3, 64, 64, 64]))
    
    def load_imgs_task_01(self):
        cases = None
  
        for _, row in self.pairs_task_01.iterrows():
            case = self.nifti_loader.load(fname=DEFAULT_GROUND_TRUTH_PATH / 'task_01' / 'EASY-RESECT' / 'NIFTI' / 'Case{}'.format(row['fixed']) / 'Case{}-FLAIR-resize.nii'.format(row['fixed']))
        
            if cases is None:
                cases = case
                index = [row['fixed']]
            else:
                cases += case
                index += [row['fixed']]
                
        return DataFrame(cases, index=index)
    
    def load_imgs_task_02(self):
        cases = None
  
        for _, row in self.pairs_task_02.iterrows():
            case = self.nifti_loader.load(fname=DEFAULT_GROUND_TRUTH_PATH / 'task_02' / 'training' / 'lungMasks' / 'case_{:03d}_exp.nii.gz'.format(row['fixed']))
        
            if cases is None:
                cases = case
                index = [row['fixed']]
            else:
                cases += case
                index += [row['fixed']]
                
        return DataFrame(cases, index=index)
    
    def load_segs_task_03(self):
        cases = None
        
        indices = []
        for _, row in self.pairs_task_03.iterrows():
            indices.append(row['fixed'])
            indices.append(row['moving'])
        indices = np.array(indices)
        
        for i in np.unique(indices):
            case = self.nifti_loader.load(fname=DEFAULT_GROUND_TRUTH_PATH / 'task_03' / 'Training' / 'label' / 'label{:04d}.nii.gz'.format(i))
        
            if cases is None:
                cases = case
                index = [i]
            else:
                cases += case
                index += [i]
                
        return DataFrame(cases, index=index)
    
    def load_segs_task_04(self):
        cases = None
        
        indices = []
        for _, row in self.pairs_task_04.iterrows():
            indices.append(row['fixed'])
            indices.append(row['moving'])
        indices = np.array(indices)
        
        for i in np.unique(indices):
            case = self.nifti_loader.load(fname=DEFAULT_GROUND_TRUTH_PATH / 'task_04' / 'Training' / 'label' / 'hippocampus_{}.nii.gz'.format(i))
        
            if cases is None:
                cases = case
                index = [i]
            else:
                cases += case
                index += [i]
                
        return DataFrame(cases, index=index)
    
    def load_lms_task_01(self):
        cases = None
        
        for _, row in self.pairs_task_01.iterrows():
            case = self.curious_lms_loader.load(fname=DEFAULT_GROUND_TRUTH_PATH / 'task_01' / 'EASY-RESECT' / 'landmarks' / 'Coordinates' / 'Case{}-MRI-beforeUS.tag'.format(row['fixed']))
        
            if cases is None:
                cases = [case]
                index = [row['fixed']]
            else:
                cases += [case]
                index += [row['fixed']]
                
        return DataFrame(cases, index=index)
    
    def load_lms_task_02(self):
        cases = None
        
        for _, row in self.pairs_task_02.iterrows():
            case = self.l2r_lms_loader.load(fname=DEFAULT_GROUND_TRUTH_PATH / 'task_02' / 'keypoints' / 'case_{:03d}.csv'.format(row['fixed']))
        
            if cases is None:
                cases = [case]
                index = [row['fixed']]
            else:
                cases += [case]
                index += [row['fixed']]
                
        return DataFrame(cases, index=index)
    
    def merge_ground_truth_and_predictions_task_01(self):
        cases = []
        for _, row in self.pairs_task_01.iterrows():
            case = {'img' : self.imgs_task_01.loc[row['fixed']],
                    'lms_fixed' : self.lms_task_01.loc[row['fixed']]['lms_fixed'],
                    'lms_moving' : self.lms_task_01.loc[row['moving']]['lms_moving'],
                    'disp_field' : self.disp_fields_task_01.loc[(row['fixed'], row['moving'])]}
            cases += [case]
        self.cases_task_01 = DataFrame(cases)
        
    def merge_ground_truth_and_predictions_task_02(self):
        cases = []
        for _, row in self.pairs_task_02.iterrows():
            case = {'img' : self.imgs_task_02.loc[row['fixed']],
                    'lms_fixed' : self.lms_task_02.loc[row['fixed']]['lms_fixed'],
                    'lms_moving' : self.lms_task_02.loc[row['moving']]['lms_moving'],
                    'disp_field' : self.disp_fields_task_02.loc[(row['fixed'], row['moving'])]}
            cases += [case]
        self.cases_task_02 = DataFrame(cases)
        
    def merge_ground_truth_and_predictions_task_03(self):
        cases = []
        for _, row in self.pairs_task_03.iterrows():
            case = {'seg_fixed' : self.segs_task_03.loc[row['fixed']],
                    'seg_moving' : self.segs_task_03.loc[row['moving']],
                    'disp_field' : self.disp_fields_task_03.loc[(row['fixed'], row['moving'])]}
            cases += [case]
        self.cases_task_03 = DataFrame(cases)
        
    def merge_ground_truth_and_predictions_task_04(self):
        cases = []
        for _, row in self.pairs_task_04.iterrows():
            case = {'seg_fixed' : self.segs_task_04.loc[row['fixed']],
                    'seg_moving' : self.segs_task_04.loc[row['moving']],
                    'disp_field' : self.disp_fields_task_04.loc[(row['fixed'], row['moving'])]}
            cases += [case]
        self.cases_task_04 = DataFrame(cases)
        
    def score_task_01(self):
        self.cases_results_task_01 = DataFrame()
        for idx, case in self.cases_task_01.iterrows():
            self.cases_results_task_01 = self.cases_results_task_01.append(self.score_case_task_01(idx=idx, case=case), ignore_index=True)

        self.aggregate_results_task_01 = self.score_aggregates_task_01()
        
    def score_task_02(self):
        self.cases_results_task_02 = DataFrame()
        for idx, case in self.cases_task_02.iterrows():
            self.cases_results_task_02 = self.cases_results_task_02.append(self.score_case_task_02(idx=idx, case=case), ignore_index=True)

        self.aggregate_results_task_02 = self.score_aggregates_task_02()
        
    def score_task_03(self):
        self.cases_results_task_03 = DataFrame()
        for idx, case in self.cases_task_03.iterrows():
            self.cases_results_task_03 = self.cases_results_task_03.append(self.score_case_task_03(idx=idx, case=case), ignore_index=True)

        self.aggregate_results_task_03 = self.score_aggregates_task_03()
        
    def score_task_04(self):
        self.cases_results_task_04 = DataFrame()
        for idx, case in self.cases_task_04.iterrows():
            self.cases_results_task_04 = self.cases_results_task_04.append(self.score_case_task_04(idx=idx, case=case), ignore_index=True)

        self.aggregate_results_task_04 = self.score_aggregates_task_04()
        
    def score_case_task_01(self, *, idx, case):
        img_path = case['img']['path']
        disp_field_path = case['disp_field']['path']
        
        img = self.nifti_loader.load_image(img_path)
        affine = img.affine
        spacing = img.header.get_zooms()
        
        disp_field = self.numpy_loader.load_image(disp_field_path).astype('float32')
        disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])
        
        lms_fixed = np.dot(np.linalg.inv(affine), np.concatenate((np.array(case['lms_fixed']), np.ones((len(case['lms_fixed']), 1))), axis=1).transpose()).transpose()[:,:3]
        lms_moving = np.dot(np.linalg.inv(affine), np.concatenate((np.array(case['lms_moving']), np.ones((len(case['lms_moving']), 1))), axis=1).transpose()).transpose()[:,:3]

        jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
        log_jac_det = np.log(jac_det)
        
        lms_fixed_disp_x = map_coordinates(disp_field[0], lms_fixed.transpose())
        lms_fixed_disp_y = map_coordinates(disp_field[1], lms_fixed.transpose())
        lms_fixed_disp_z = map_coordinates(disp_field[2], lms_fixed.transpose())
        lms_fixed_disp = np.array((lms_fixed_disp_x, lms_fixed_disp_y, lms_fixed_disp_z)).transpose()
       
        lms_fixed_warped = lms_fixed + lms_fixed_disp

        tre = compute_tre(lms_fixed_warped, lms_moving, spacing)
   
        return {'TRE' : tre.mean(),
                'LogJacDetStd' : log_jac_det.std()}

    def score_case_task_02(self, *, idx, case):
        img_path = case['img']['path']
        disp_field_path = case['disp_field']['path']
        
        img = self.nifti_loader.load_image(img_path)
        spacing = img.header.get_zooms()
        
        disp_field = self.numpy_loader.load_image(disp_field_path).astype('float32')
        disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])
        
        lms_fixed = np.array(case['lms_fixed'])
        lms_moving = np.array(case['lms_moving'])

        jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
        log_jac_det = np.log(jac_det)
        
        lms_fixed_disp_x = map_coordinates(disp_field[0], lms_fixed.transpose())
        lms_fixed_disp_y = map_coordinates(disp_field[1], lms_fixed.transpose())
        lms_fixed_disp_z = map_coordinates(disp_field[2], lms_fixed.transpose())
        lms_fixed_disp = np.array((lms_fixed_disp_x, lms_fixed_disp_y, lms_fixed_disp_z)).transpose()
       
        lms_fixed_warped = lms_fixed + lms_fixed_disp

        tre = compute_tre(lms_fixed_warped, lms_moving, spacing)
        
        return {'TRE' : tre.mean(),
                'LogJacDetStd' : np.ma.MaskedArray(log_jac_det, 1-img.get_fdata()[2:-2, 2:-2, 2:-2]).std()}

    def score_case_task_03(self, *, idx, case):
        fixed_path = case['seg_fixed']['path']
        moving_path = case['seg_moving']['path']
        disp_field_path = case['disp_field']['path']
        
        fixed = self.nifti_loader.load_image(fixed_path).get_fdata()
        spacing = self.nifti_loader.load_image(fixed_path).header.get_zooms()
        moving = self.nifti_loader.load_image(moving_path).get_fdata()
        disp_field = self.numpy_loader.load_image(disp_field_path).astype('float32')
        disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])
       
        jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
        log_jac_det = np.log(jac_det)
        
        D, H, W = fixed.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        moving_warped = map_coordinates(moving, identity + disp_field, order=0)
        
        # dice
        dice = 0
        count = 0
        for i in range(1, 14):
            if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
                continue
            dice += compute_dice_coefficient((fixed==i), (moving_warped==i))
            count += 1
        dice /= count
        
        # hd95
        hd95 = 0
        count = 0
        for i in range(1, 14):
            if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
                continue
            hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), np.ones(3)), 95.)
            count += 1
        hd95 /= count
        
        return {'DiceCoefficient' : dice,
                'HausdorffDistance95' : hd95,
                'LogJacDetStd' : log_jac_det.std()}
    
    def score_case_task_04(self, *, idx, case):
        fixed_path = case['seg_fixed']['path']
        moving_path = case['seg_moving']['path']
        disp_field_path = case['disp_field']['path']
        
        fixed = self.nifti_loader.load_image(fixed_path).get_fdata()
        spacing = self.nifti_loader.load_image(fixed_path).header.get_zooms()
        moving = self.nifti_loader.load_image(moving_path).get_fdata()
        disp_field = self.numpy_loader.load_image(disp_field_path).astype('float32')
       
        jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
        log_jac_det = np.log(jac_det)
        
        D, H, W = fixed.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        moving_warped = map_coordinates(moving, identity + disp_field, order=0)
        
        # dice
        dice = 0
        count = 0
        for i in range(1, 3):
            if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
                continue
            dice += compute_dice_coefficient((fixed==i), (moving_warped==i))
            count += 1
        dice /= count
        
        # hd95
        hd95 = 0
        count = 0
        for i in range(1, 3):
            if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
                continue
            hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), np.ones(3)), 95.)
            count += 1
        hd95 /= count
        
        return {'DiceCoefficient' : dice,
                'HausdorffDistance95' : hd95,
                'LogJacDetStd' : log_jac_det.std()}

    def score_aggregates_task_01(self):
        aggregate_results = {}

        for col in self.cases_results_task_01.columns:
            aggregate_results[col] = self.aggregate_series_task_01(series=self.cases_results_task_01[col])

        return aggregate_results
    
    def score_aggregates_task_02(self):
        aggregate_results = {}

        for col in self.cases_results_task_02.columns:
            aggregate_results[col] = self.aggregate_series_task_02(series=self.cases_results_task_02[col])

        return aggregate_results
    
    def score_aggregates_task_03(self):
        aggregate_results = {}

        for col in self.cases_results_task_03.columns:
            aggregate_results[col] = self.aggregate_series_task_03(series=self.cases_results_task_03[col])

        return aggregate_results
    
    def score_aggregates_task_04(self):
        aggregate_results = {}

        for col in self.cases_results_task_04.columns:
            aggregate_results[col] = self.aggregate_series_task_04(series=self.cases_results_task_04[col])

        return aggregate_results
    
    def aggregate_series_task_01(self, *, series):
        series_summary = {}
        
        series_summary['mean'] = series.mean()
        series_summary['std'] = series.std()
        
        return series_summary
    
    def aggregate_series_task_02(self, *, series):
        series_summary = {}
        
        series_summary['mean'] = series.mean()
        series_summary['std'] = series.std()
        
        return series_summary
    
    def aggregate_series_task_03(self, *, series):
        series_summary = {}
        
        series_summary['mean'] = series.mean()
        series_summary['std'] = series.std()
        series_summary['30'] = series.quantile(.3)
        
        return series_summary
    
    def aggregate_series_task_04(self, *, series):
        series_summary = {}
        
        series_summary['mean'] = series.mean()
        series_summary['std'] = series.std()
        series_summary['30'] = series.quantile(.3)
        
        return series_summary
        
    def load_pairs(self, fname):
        return DataFrame(self.csv_loader.load(fname=fname))
    
    def load_disp_fields(self, pairs, folder, expected_shape):
        cases = None
        
        for _, row in pairs.iterrows():
            fname = folder / 'disp_{:04d}_{:04d}.npz'.format(row['fixed'], row['moving'])
            
            if os.path.isfile(fname):
                case = self.numpy_loader.load(fname=fname)
                
                disp_field = self.numpy_loader.load_image(fname=fname)
                dtype = disp_field.dtype
                if not dtype == 'float16':
                    raise_dtype_error(fname, dtype)
                    
                shape = np.array(disp_field.shape)
                if not (shape==expected_shape).all():
                    raise_shape_error(fname, shape, expected_shape)

                if cases is None:
                    cases = case
                    index = [(row['fixed'], row['moving'])]
                else:
                    cases += case
                    index.append((row['fixed'], row['moving']))
            else:
                raise_missing_file_error(fname)
                
        return DataFrame(cases, index=MultiIndex.from_tuples(index))
    
    def metrics(self):
        return {"task_01" : {
                    "case": self.cases_results_task_01.to_dict(),
                    "aggregates": self.aggregate_results_task_01,
                            },
                "task_02" : {
                    "case": self.cases_results_task_02.to_dict(),
                    "aggregates": self.aggregate_results_task_02,
                            },
                "task_03" : {
                    "case": self.cases_results_task_03.to_dict(),
                    "aggregates": self.aggregate_results_task_03,
                            },
                "task_04" : {
                    "case": self.cases_results_task_04.to_dict(),
                    "aggregates": self.aggregate_results_task_04,
                            }
               }
    
    def save(self):
        with open(self.output_file, "w") as f:
            f.write(json.dumps(self.metrics()))
            
##### main #####
    
if __name__ == "__main__":
    EvalVal().evaluate()
