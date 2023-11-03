from torch.utils import data

import torch
import random
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from scipy.ndimage import affine_transform
from torchvision.transforms import ToTensor

class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch
    # Takes a bunch of arguments for data augmentation as well
    def __init__(self, fixed_paths, moving_paths, rotate=None, shear=None, translate=None):
        assert (len (moving_paths) == len (fixed_paths))

        self.moving_paths = moving_paths
        self.moving_paths.sort ()

        self.fixed_paths = fixed_paths
        self.fixed_paths.sort ()

        self.rotate = rotate
        self.shear = shear
        self.translate = translate
        # self.mask_paths = mask_paths
        # self.mask_paths.sort ()


    def __len__(self):
        return len(self.moving_paths)

    def __getitem__(self, index):
        # Generates one sample of data
        moving_path = self.moving_paths[index]
        fixed_path = self.fixed_paths[index]
        #mask_path = self.mask_paths[index]

        sitk_fixed = sitk.ReadImage(fixed_path)
        fixed = sitk.GetArrayFromImage(sitk_fixed)

        sitk_moving = sitk.ReadImage(moving_path)
        moving = sitk.GetArrayFromImage(sitk_moving)
        aug_moving, transform_mat = self.__augment_image__(moving)

        return fixed, moving, aug_moving, transform_mat

    def __augment_image__(self, data):
        # Rotates image randomly
        np.set_printoptions(suppress=True)

        match self.rotate:
            case ((lower_x, upper_x), (lower_y, upper_y), (lower_z, upper_z)):
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
            case None:
                rot_mat = torch.eye (4, dtype=torch.double)
        match self.shear:
            case ((lower_x, upper_x), (lower_y, upper_y), (lower_z, upper_z)):
                s_x = random.uniform (lower_x, upper_x)
                s_y = random.uniform (lower_y, upper_y)
                s_z = random.uniform (lower_z, upper_z)
                shear_mat = torch.tensor ([[ 1,   s_y, s_z, 0],
                                           [ s_x, 1,   s_z, 0],
                                           [ s_x, s_y, 1,   0],
                                           [ 0,   0,   0,   1]],
                                          dtype=torch.double)
            case None:
                shear_mat = torch.eye (4, dtype=torch.double)
        match self.translate:
            case ((lower_x, upper_x), (lower_y, upper_y), (lower_z, upper_z)):
                transl_mat = torch.tensor ([[1, 0, 0, random.uniform (lower_x, upper_x)],
                                            [0, 1, 0, random.uniform (lower_y, upper_y)],
                                            [0, 0, 1, random.uniform (lower_z, upper_z)],
                                            [0, 0, 0, 1]],
                                           dtype=torch.double)
            case None:
                transl_mat = torch.eye (4, dtype=torch.double)

        # extra translation to make rotation around center
        c_in = 0.5*torch.tensor (data.shape,dtype=torch.double)
        offset = c_in - torch.matmul (c_in, rot_mat[:-1,:-1])
        transl_mat[:-1,-1] = -offset

        transform_mat = torch.eye (4, dtype=torch.double)
        transform_mat = torch.matmul (transform_mat, shear_mat)
        transform_mat = torch.matmul (transform_mat, rot_mat)
        transform_mat = torch.matmul (transform_mat, transl_mat)

        return affine_transform (data, transform_mat, mode="nearest"), transform_mat
