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
    data = affine_transform (image.dataobj, transform, mode=mode, order=0)
    nifti_affine = np.matmul (image.affine, transform)
    return nib.Nifti1Image (data, nifti_affine)

class SynthradData (data.Dataset):
    def __find_files__ (self, path):
        ret_val = [ path + img for img in os.listdir (path) ]
        ret_val.sort()
        return ret_val

    # Characterizes a dataset for PyTorch
    def __init__(self, img_folder, fixed_cts=None, fixed_mrs=None, moving_cts=None, moving_mrs=None, include_mask=True):
        if fixed_cts is None:
            self.fixed_cts = self.__find_files__(img_folder + "fixed/ct/")
        else:
            self.fixed_cts = self.__find_files__(fixed_cts)
        if fixed_mrs is None:
            self.fixed_mrs = self.__find_files__(img_folder + "fixed/mr/")
        else:
            self.fixed_mrs = self.__find_files__(fixed_mrs)

        if moving_cts is None:
            self.moving_cts = self.__find_files__(img_folder + "moving/ct/")
        else:
            self.moving_cts = self.__find_files__(moving_cts)
        if moving_mrs is None:
            self.moving_mrs = self.__find_files__(img_folder + "moving/mr/")
        else:
            self.moving_mrs = self.__find_files__(moving_mrs)

        if include_mask:
            self.fixed_masks = self.__find_files__(img_folder + "fixed/mask/")
            self.moving_masks = self.__find_files__(img_folder + "moving/mask/")

        self.transforms = self.__find_files__(img_folder + "transforms/")

        self.include_mask = include_mask

    def __len__(self):
        return len(self.fixed_cts)

    # Generates one sample of data
    def __getitem__(self, index):
        # Removing the .affine to world coordinates for use with pytorch dataloader.
        # Probably not super good, but augmentation works properly with it
        ct_fixed = np.expand_dims (nib.load (self.fixed_cts[index]).get_fdata(), axis=-1)
        mr_fixed = np.expand_dims (nib.load (self.fixed_mrs[index]).get_fdata(), axis=-1)

        ct_moving = nib.load (self.moving_cts[index])
        mr_moving = nib.load (self.moving_mrs[index])

        inv_transform = ct_moving.affine # Doesn't matter which moving image it's from

        ct_moving = np.expand_dims (ct_moving.get_fdata(), axis=-1)
        mr_moving = np.expand_dims (mr_moving.get_fdata(), axis=-1)

        transform = torch.load (self.transforms[index])

        if self.include_mask:
            mask_fixed = np.expand_dims (nib.load (self.fixed_masks[index]).get_fdata(), axis=-1)
            mask_moving = np.expand_dims (nib.load (self.moving_masks[index]).get_fdata(), axis=-1)

            return ct_fixed, mr_fixed, mask_fixed, ct_moving, mr_moving, mask_moving, transform, inv_transform
        else:
            return ct_fixed, mr_fixed, ct_moving, mr_moving, transform, inv_transform

class AugmentData (data.Dataset):
    # Characterizes a dataset for PyTorch
    # Takes a bunch of arguments for data augmentation as well
    def __init__(self, ct_paths, mr_paths, mask_paths, rotate=None, shear=None, translate=None, pad_to=None, normalise=False, side_view=False):
        assert (len (mr_paths) == len (ct_paths))

        self.ct_paths = ct_paths
        self.ct_paths.sort ()

        self.mr_paths = mr_paths
        self.mr_paths.sort ()

        self.mask_paths = mask_paths
        self.mask_paths.sort ()

        self.rotate = rotate
        self.shear = shear
        self.translate = translate
        self.pad_to = pad_to
        self.normalise = normalise
        self.side_view = side_view

    def __max_size__(self):
        size = [0,0,0]
        for img in self.ct_paths:
            img_size = nib.load(img).get_fdata().shape
            size = [ max (a,b) for a, b in zip (size, img_size) ]
        return size

    def __len__(self):
        return len(self.mr_paths)

    def __pad_center__(self, image, pad_size):
        pads = [ a-b for a,b in zip (pad_size, image.shape) ]
        return np.pad (image,
                       ((pads[0]//2, ceil(pads[0]/2)),
                        (pads[1]//2, ceil(pads[1]/2)),
                        (pads[2]//2, ceil(pads[2]/2))),
                       mode="constant",
                       constant_values=0)

    # Generates one sample of data
    def __getitem__(self, index):
        ct_path = self.ct_paths[index]
        mr_path = self.mr_paths[index]
        mask_path = self.mask_paths[index]

        # Removing the .affine to world coordinates for use with pytorch dataloader.
        # Probably not super good, but augmentation works properly with it
        ct_fixed = nib.load (ct_path).get_fdata()
        mr_fixed = nib.load (mr_path).get_fdata()
        mask = nib.load (mask_path).get_fdata()

        if self.normalise:
            ct_fixed = normalise_array (ct_fixed)
            mr_fixed = normalise_array (mr_fixed)

        if self.side_view:
            ct_fixed = np.flip (np.transpose(ct_fixed, [2,1,0]), axis=0)
            mr_fixed = np.flip (np.transpose(mr_fixed, [2,1,0]), axis=0)

        if self.pad_to is not None:
            ct_fixed = self.__pad_center__(ct_fixed, self.pad_to)
            mr_fixed = self.__pad_center__(mr_fixed, self.pad_to)
            mask = self.__pad_center__(mask, self.pad_to)

        mr_moving, transform_mat, inv_transform = self.__augment_image__(mr_fixed)
        ct_moving = affine_transform (ct_fixed, transform_mat, order=0)
        aug_mask = affine_transform (mask, transform_mat, order=0)

        return ct_fixed, mr_fixed, mask, ct_moving, mr_moving, aug_mask, transform_mat, inv_transform

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
        ret_val = affine_transform (data, transform_mat, order=0)
        return ret_val, transform_mat, inv_transform_mat
