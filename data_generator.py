import os
import torch
import random
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch.nn.functional as F

from PIL import Image
from math import ceil
from torch.utils import data
from scipy.ndimage import affine_transform
from torchvision.transforms import ToTensor

# Normalisation between 0 and 1
def normalise_array (arr):
    min_val = np.min (arr)
    max_val = np.max (arr)
    return (arr-min_val) / (max_val-min_val)

def nifti_affine (image, transform, mode="nearest"):
    data = affine_transform (image.dataobj, transform, mode=mode)
    nifti_affine = np.matmul (image.affine, transform)
    return nib.Nifti1Image (data, nifti_affine)

class SynthradData (data.Dataset):
    def __find_files__ (self, path, ident):
        ret_val = [ path + img for img in os.listdir (path) if ident in img ]
        ret_val.sort()
        return ret_val

    # Characterizes a dataset for PyTorch
    def __init__(self, img_folder, include_original=False, include_mask=True):
        self.fixed_ct = self.__find_files__ (img_folder, "_ct.nii.gz")
        self.fixed_masks = self.__find_files__ (img_folder, "_fixed_mask.nii.gz")
        self.aug_mr = self.__find_files__ (img_folder, "_aug_mr.nii.gz")
        self.aug_masks = self.__find_files__ (img_folder, "_aug_mask.nii.gz")
        self.orig_mr = self.__find_files__ (img_folder, "_orig_mr.nii.gz")
        self.transforms = self.__find_files__ (img_folder, ".pt")

        self.include_original = include_original
        self.include_mask = include_mask

    def __len__(self):
        return len(self.fixed_ct)

    # Generates one sample of data
    def __getitem__(self, index):
        # Removing the .affine to world coordinates for use with pytorch dataloader.
        # Probably not super good, but augmentation works properly with it
        fixed = np.expand_dims (nib.load (self.fixed_ct[index]).get_fdata(), axis=-1)

        moving = nib.load (self.aug_mr[index])
        inv_transform = moving.affine
        moving = np.expand_dims (moving.get_fdata(), axis=-1)
        transform = torch.load (self.transforms[index])

        if self.include_mask:
            fixed_mask = np.expand_dims (nib.load (self.fixed_masks[index]).get_fdata(), axis=-1)
            moving_mask = np.expand_dims (nib.load (self.aug_masks[index]).get_fdata(), axis=-1)

            if self.include_original:
                orig_moving = np.expand_dims (nib.load (self.orig_mr[index]).get_fdata(), axis=-1)
                return fixed, fixed_mask, moving, moving_mask, orig_moving, transform, inv_transform
            else:
                return fixed, fixed_mask, moving, moving_mask, transform, inv_transform
        else:
            if self.include_original:
                orig_moving = np.expand_dims (nib.load (self.orig_mr[index]).get_fdata(), axis=-1)
                return fixed, moving, orig_moving, transform, inv_transform
            else:
                return fixed, moving, transform, inv_transform

class AugmentData (data.Dataset):
    # Characterizes a dataset for PyTorch
    # Takes a bunch of arguments for data augmentation as well
    def __init__(self, fixed_paths, moving_paths, mask_paths, rotate=None, shear=None, translate=None, pad_to=None, normalise=False):
        assert (len (moving_paths) == len (fixed_paths))

        self.fixed_paths = fixed_paths
        self.fixed_paths.sort ()

        self.moving_paths = moving_paths
        self.moving_paths.sort ()

        self.mask_paths = mask_paths
        self.mask_paths.sort ()

        self.rotate = rotate
        self.shear = shear
        self.translate = translate
        self.pad_to = pad_to
        self.normalise = normalise

    def __max_size__(self):
        size = [0,0,0]
        for img in self.fixed_paths:
            img_size = nib.load(img).get_fdata().shape
            size = [ max (a,b) for a, b in zip (size, img_size) ]
        return size

    def __len__(self):
        return len(self.moving_paths)

    def __pad_center__(self, image, pad_size):
        pads = [ a-b for a,b in zip (pad_size, image.shape) ]
        return np.pad (image,
                       ((pads[0]//2, ceil(pads[0]/2)),
                        (pads[1]//2, ceil(pads[1]/2)),
                        (pads[2]//2, ceil(pads[2]/2))),
                       mode="constant",
                       constant_values=0)

    def __getitem__(self, index):
        # Generates one sample of data
        fixed_path = self.fixed_paths[index]
        moving_path = self.moving_paths[index]
        mask_path = self.mask_paths[index]

        # Removing the .affine to world coordinates for use with pytorch dataloader.
        # Probably not super good, but augmentation works properly with it
        fixed = nib.load (fixed_path).get_fdata()
        moving = nib.load (moving_path).get_fdata()
        mask = nib.load (mask_path).get_fdata()

        if self.normalise:
            fixed = normalise_array (fixed)
            moving = normalise_array (moving)

        if self.pad_to is not None:
            fixed = self.__pad_center__(fixed, self.pad_to)
            moving = self.__pad_center__(moving, self.pad_to)
            mask = self.__pad_center__(mask, self.pad_to)

        aug_moving, transform_mat, inv_transform = self.__augment_image__(moving)
        aug_mask = affine_transform (mask, transform_mat)

        return fixed, mask, moving, aug_moving, aug_mask, transform_mat, inv_transform

    def __augment_image__(self, data):
        # Rotates nifti image randomly
        np.set_printoptions(suppress=True)
        #data = image.get_fdata()

        if self.rotate is not None:
            if type (self.rotate) is float or type (self.rotate) is int:
                (lower_x, upper_x) = -self.rotate, self.rotate
                (lower_y, upper_y) = -self.rotate, self.rotate
                (lower_z, upper_z) = -self.rotate, self.rotate
            else:
                ((lower_x, upper_x), (lower_y, upper_y), (lower_z, upper_z)) = self.rotate

            rot_x = random.uniform (lower_x, upper_x)
            rot_y = random.uniform (lower_y, upper_y)
            #rot_y = np.pi/2
            rot_z = random.uniform (lower_z, upper_z)


            rot_x_mat = torch.tensor ([[ 1,             0,              0,              0],
                                       [ 0,             np.cos(rot_x),  np.sin(rot_x),  0],
                                       [ 0,             -np.sin(rot_x), np.cos(rot_x),  0],
                                       [ 0,             0,              0,              1]],
                                      dtype=torch.double)
            rot_y_mat = torch.tensor ([[ np.cos(rot_y), 0,              -np.sin(rot_y), 0],
                                       [ 0,             1,              0,              0],
                                       [ np.sin(rot_y), 0,              np.cos(rot_y),  0],
                                       [ 0,             0,              0,              1]],
                                      dtype=torch.double)
            rot_z_mat = torch.tensor ([[ np.cos(rot_z), -np.sin(rot_z), 0,              0],
                                       [ np.sin(rot_z), np.cos(rot_z),  0,              0],
                                       [ 0,             0,              1,              0],
                                       [ 0,             0,              0,              1]],
                                      dtype=torch.double)

            rot_mat = torch.matmul (rot_x_mat, rot_y_mat)
            rot_mat = torch.matmul (rot_mat, rot_z_mat)

        else:
            rot_mat = torch.eye (4, dtype=torch.double)

        if self.shear is not None:
            if type (self.shear) is float or type (self.shear) is int:
                (lower_x, upper_x) = -self.shear, self.shear
                (lower_y, upper_y) = -self.shear, self.shear
                (lower_z, upper_z) = -self.shear, self.shear
            else:
                ((lower_x, upper_x), (lower_y, upper_y), (lower_z, upper_z)) = self.shear

            s_x = random.uniform (lower_x, upper_x)
            s_y = random.uniform (lower_y, upper_y)
            s_z = random.uniform (lower_z, upper_z)
            shear_mat = torch.tensor ([[ 1,   s_y, s_z, 0],
                                       [ s_x, 1,   s_z, 0],
                                       [ s_x, s_y, 1,   0],
                                       [ 0,   0,   0,   1]],
                                      dtype=torch.double)
        else:
            shear_mat = torch.eye (4, dtype=torch.double)

        if self.translate is not None:
            if type (self.translate) is float or type (self.translate) is int:
                (lower_x, upper_x) = -self.translate, self.translate
                (lower_y, upper_y) = -self.translate, self.translate
                (lower_z, upper_z) = -self.translate, self.translate
            else:
                ((lower_x, upper_x), (lower_y, upper_y), (lower_z, upper_z)) = self.translate

            transl_mat = torch.tensor ([[1, 0, 0, random.uniform (lower_x, upper_x)],
                                        [0, 1, 0, random.uniform (lower_y, upper_y)],
                                        [0, 0, 1, random.uniform (lower_z, upper_z)],
                                        [0, 0, 0, 1]],
                                       dtype=torch.double)
        else:
            transl_mat = torch.eye (4, dtype=torch.double)

        # Extra translation to make rotation around center
        c_in = 0.5*torch.tensor (data.shape,dtype=torch.double)
        offset = c_in - torch.matmul (c_in, rot_mat[:-1,:-1])
        transl_mat[:-1,-1] -= offset

        # Final transformation matrix
        transform_mat = torch.eye (4, dtype=torch.double)
        transform_mat = torch.matmul (transform_mat, shear_mat)
        transform_mat = torch.matmul (transform_mat, rot_mat)
        transform_mat = torch.matmul (transform_mat, transl_mat)

        # Inverse affine transformation
        inv_transform_mat = torch.eye(4, dtype=torch.double)
        # inverse rotation
        inv_transform_mat[:3,:3] = torch.transpose (transform_mat[:3,:3], 0, 1)
        # inverse translation
        inv_transform_mat[:-1,-1] = torch.matmul (torch.transpose (transform_mat[:3,:3], 0, 1), -transform_mat[:-1,-1])

        #ret_val = nifti_affine (image, transform_mat)
        ret_val = affine_transform (data, transform_mat)
        return ret_val, transform_mat, inv_transform_mat
