from evalutils.io import CSVLoader, FileLoader, ImageLoader
import numpy as np
import nibabel as nib

class NiftiLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        nii = nib.load(str(fname))
        return nii.get_data(), nii.header.get_zooms()

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
