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
from nibabel.orientations import axcodes2ornt, io_orientation, apply_orientation, inv_ornt_aff

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
        self.inv_transforms = self.__find_files__(img_folder + "inv_transforms/")

        self.include_mask = include_mask


    def __len__(self):
        return len(self.fixed_cts)

    def shape (self):
        return nib.load (self.moving_cts[0]).shape

    # Generates one sample of data
    def __getitem__(self, index):
        # Removing the .affine to world coordinates for use with pytorch dataloader.
        #ct_fixed = np.expand_dims (nib.load (self.fixed_cts[index]).get_fdata(), axis=-1)
        #mr_fixed = np.expand_dims (nib.load (self.fixed_mrs[index]).get_fdata(), axis=-1)
        ct_fixed = nib.load (self.fixed_cts[index])
        mr_fixed = nib.load (self.fixed_mrs[index])

        ct_moving = nib.load (self.moving_cts[index])
        mr_moving = nib.load (self.moving_mrs[index])


        #ct_moving = np.expand_dims (ct_moving.get_fdata(), axis=-1)
        #mr_moving = np.expand_dims (mr_moving.get_fdata(), axis=-1)

        transform = torch.load (self.transforms[index])
        inv_transform = torch.load (self.inv_transforms[index])

        if self.include_mask:
            mask_fixed = nib.load (self.fixed_masks[index])
            mask_moving = nib.load (self.moving_masks[index])

            return ct_fixed, mr_fixed, mask_fixed, ct_moving, mr_moving, mask_moving, transform, inv_transform
        else:
            return ct_fixed, mr_fixed, ct_moving, mr_moving, transform, inv_transform

# Won't work with a dataloader, because I would neet to implement a collate_fn function for nifti files
class AugmentData (data.Dataset):
    # Characterizes a dataset for PyTorch
    # Takes a bunch of arguments for data augmentation as well
    def __init__(self, ct_paths, mr_paths, mask_paths, rotate=None, shear=None, translate=None, pad_to=None, normalise=False, orientation=None, aug_path=None, transform_fixed=False):
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
        self.normalise = normalise
        self.orientation = orientation
        self.aug_path = aug_path
        self.transform_fixed = transform_fixed

        if pad_to is not None:
            self.pad_to = pad_to
        else:
            self.pad_to = self.__max_size__()
        print (self.pad_to)

    def __apply_orientation__ (self, nib_img, orientation):
        new_or = axcodes2ornt (orientation)
        old_or = io_orientation (nib_img.affine)
        if (new_or == old_or).all(): return nib_img

        transform = nib.orientations.ornt_transform(old_or, new_or)
        data = apply_orientation (nib_img.get_fdata(), transform)
        new_affine = nib_img.affine @ inv_ornt_aff (transform, data.shape)
        ret_val = nib.Nifti1Image (data, new_affine, header=nib_img.header)
        return ret_val

    def __max_size__(self):
        size = [0,0,0]
        for img in self.ct_paths:
            img_size = nib.load(img).shape
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
        ct_fixed_nib = nib.load (ct_path)
        mr_fixed_nib = nib.load (mr_path)
        mask_nib = nib.load (mask_path)

        if self.orientation is not None:
            ct_fixed_nib = self.__apply_orientation__ (ct_fixed_nib, self.orientation)
            mr_fixed_nib = self.__apply_orientation__ (mr_fixed_nib, self.orientation)
            mask_nib = self.__apply_orientation__ (mask_nib, self.orientation)

        ct_fixed = ct_fixed_nib.get_fdata()
        mr_fixed = mr_fixed_nib.get_fdata()
        mask = mask_nib.get_fdata()

        if self.normalise:
            ct_fixed = normalise_array (ct_fixed)
            mr_fixed = normalise_array (mr_fixed)

        if self.pad_to:
            ct_fixed = self.__pad_center__(ct_fixed, self.pad_to)
            mr_fixed = self.__pad_center__(mr_fixed, self.pad_to)
            mask = self.__pad_center__(mask, self.pad_to)

            ct_fixed_nib = nib.Nifti1Image (ct_fixed,
                                            ct_fixed_nib.affine,
                                            header=ct_fixed_nib.header)
            ct_fixed_nib.header["dim"][1:4] = self.pad_to
            mr_fixed_nib = nib.Nifti1Image (mr_fixed,
                                            mr_fixed_nib.affine,
                                            header=mr_fixed_nib.header)
            mr_fixed_nib.header["dim"][1:4] = self.pad_to
            mask_nib = nib.Nifti1Image (mask,
                                        mask_nib.affine,
                                        header=mask_nib.header)
            mask_nib.header["dim"][1:4] = self.pad_to

        mr_moving, transform_mat, inv_transform = self.__augment_image__(mr_fixed)
        ct_moving = affine_transform (ct_fixed, transform_mat, order=0)
        aug_mask = affine_transform (mask, transform_mat, order=0)

        # Option to transform the fixed scan as well,
        # to prevent fine-tuning from just learning to go to the middle
        if self.transform_fixed:
            mr_fixed, fixed_transform_mat, fixed_inv_transform = self.__augment_image__(mr_fixed)
            ct_fixed = affine_transform (ct_fixed, fixed_transform_mat, order=0)
            aug_mask = affine_transform (mask, fixed_transform_mat, order=0)

            mr_aff = mr_fixed_nib.affine @ fixed_transform_mat.numpy()
            mr_header = mr_fixed_nib.header
            mr_header["dim"][1:4] = mr_moving.shape
            mr_fixed_nib = nib.Nifti1Image (mr_fixed, mr_aff, header=mr_header)

            ct_aff = ct_fixed_nib.affine @ fixed_transform_mat.numpy()
            ct_header = ct_fixed_nib.header
            ct_header["dim"][1:4] = ct_moving.shape
            ct_fixed_nib = nib.Nifti1Image (ct_fixed, ct_aff, header=mr_header)

            combined_transform = fixed_inv_transform @ transform_mat
            combined_inv_transform = inv_transform @ fixed_transform_mat
        else:
            combined_transform = transform_mat
            combined_inv_transform = inv_transform

        mr_aff = mr_fixed_nib.affine @ transform_mat.numpy()
        mr_header = mr_fixed_nib.header
        mr_header["dim"][1:4] = mr_moving.shape
        mr_moving_nib = nib.Nifti1Image (mr_moving, mr_aff, header=mr_header)

        ct_aff = ct_fixed_nib.affine @ transform_mat.numpy()
        ct_header = ct_fixed_nib.header
        ct_header["dim"][1:4] = ct_moving.shape
        ct_moving_nib = nib.Nifti1Image (ct_moving, ct_aff, header=ct_header)

        mask_aff = mask_nib.affine @ transform_mat.numpy()
        mask_header = mask_nib.header
        mask_header["dim"][1:4] = aug_mask.shape
        aug_mask_nib = nib.Nifti1Image (aug_mask, mask_aff, header=mask_header)

        #if self.aug_path is not None:
        #    nib.save (ct_fixed_nib,
        #              self.aug_path + f"fixed/ct/{index}.nii.gz")
        #    nib.save (mr_fixed_nib,
        #              self.aug_path + f"fixed/mr/{index}.nii.gz")
        #    nib.save (mask_nib,
        #              self.aug_path + f"fixed/mask/{index}.nii.gz")
        #    nib.save (ct_moving_nib,
        #              self.aug_path + f"moving/ct/{index}.nii.gz")
        #    nib.save (mr_moving_nib,
        #              self.aug_path + f"moving/mr/{index}.nii.gz")
        #    nib.save (aug_mask_nib,
        #              self.aug_path + f"moving/mask/{index}.nii.gz")
        #    torch.save (transform_mat, self.aug_path + "transforms/" + f"{index}.pt")
        #    torch.save (inv_transform, self.aug_path + "inv_transforms/" + f"{index}.pt")
        return ct_fixed_nib, mr_fixed_nib, mask_nib, ct_moving_nib, mr_moving_nib, aug_mask_nib, combined_transform, combined_inv_transform

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
        #inv_transform_mat = torch.eye(4, dtype=torch.double)
        # inverse rotation
        #inv_transform_mat[:3,:3] = torch.transpose (transform_mat[:3,:3], 0, 1)
        # inverse translation
        #inv_transform_mat[:-1,-1] = torch.matmul (torch.transpose (transform_mat[:3,:3], 0, 1), -transform_mat[:-1,-1])
        inv_transform_mat = torch.linalg.inv (transform_mat)

        #ret_val = nifti_affine (image, transform_mat)
        ret_val = affine_transform (data, transform_mat, order=0)
        return ret_val, transform_mat, inv_transform_mat
