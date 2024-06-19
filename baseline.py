import os
import torch
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm
import matplotlib.pyplot as plt

from torch.utils import data
from scipy.linalg import norm
from data_generator import SynthradData
from scipy.ndimage import affine_transform

# Normalisation between 0 and 1
def normalise_array (arr):
    min_val = np.min (arr)
    max_val = np.max (arr)
    return (arr-min_val) / (max_val-min_val)

def register (model, fixed, moving, fixed_mask=None, moving_mask=None, transform=True):
    norm_fixed = normalise_array (fixed.numpy())
    norm_moving = normalise_array (moving.numpy())

    if fixed_mask is not None:
        norm_fixed = norm_fixed * fixed_mask
    if moving_mask is not None:
        norm_moving = norm_moving * moving_mask

    pred_transform = model.predict ((norm_moving, norm_fixed), verbose=0)

    if transform:
        # make_dense=False in the affine model means shift_center must be False as well
        out_img = vxm.layers.SpatialTransformer(fill_value=0,shift_center=False)((moving.numpy(), pred_transform))

    # synthmorph output just doesn't include the bottom line of the affine matrix for some reason
    pred_transform = np.concatenate ((pred_transform, [[[0,0,0,1]]]), axis=1)

    if transform:
        return out_img.numpy().astype(np.float64), pred_transform
    else:
        return pred_transform


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    num_workers = 1

    aug_path = "./aug_data/norm_rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/norm_rot0.2_trans20_shear0.1/"
    #aug_path = "./aug_data/rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/rot0.2_trans20_shear0.1/"

    # Registration model.
    in_shape = (280, 284, 262) # Shape is hardcoded for the sake of speed, but found from AugmentData.__max_size__(). Corresponds to maximal shape of all images
    model_aff = vxm.tf.networks.VxmAffineFeatureDetector(in_shape, rigid=False, make_dense=False)
    model_aff.load_weights ("synthmorph.affine.2.h5")

    aug_data = SynthradData (aug_path, include_original=True, include_mask=False)
    dataloader = data.DataLoader(aug_data,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=num_workers)

    ct_mr_inv_results = []
    mr_mr_inv_results = []

    ct_mr_abs_results = []
    mr_mr_abs_results = []
    for i, (fixed, moving, orig_moving, transform, inv_transform) in enumerate (dataloader):
        if i%50 == 0:
            print (i, flush=True)

        output_transformed = i == 0 # outputs the first transformed image for visualisation

        pred_transform_1 = register (model_aff, fixed, moving, transform=output_transformed)
        pred_transform_2 = register (model_aff, orig_moving, moving, transform=output_transformed)
        if output_transformed:
            ct_mr_result, pred_transform_1 = pred_transform_1
            mr_mr_result, pred_transform_2 = pred_transform_2

        # First error is calculated based on knowledge that the predicted transform is the inverse
        # of the one used to generate it. So transform x pred_transform should become an identity
        # matrix. The 2-norm of this is then taken to calculate error value
        ct_mr_inv_results.append (norm (np.matmul (transform, pred_transform_1) - np.eye(4)))
        mr_mr_inv_results.append (norm (np.matmul (transform, pred_transform_2) - np.eye(4)))

        # Since the inverse transformation matrix is available, I use that as well
        ct_mr_abs_results.append (norm (inv_transform - pred_transform_1))
        mr_mr_abs_results.append (norm (inv_transform - pred_transform_2))


        # Saving an image pair for visualisation
        if output_transformed:
            ct_mr_result = nib.Nifti1Image (ct_mr_result, affine=pred_transform_1.squeeze())
            mr_mr_result = nib.Nifti1Image (mr_mr_result, affine=pred_transform_2.squeeze())
            nib.save (mr_mr_result, "mr_mr_moved.nii.gz")
            nib.save (ct_mr_result, "ct_mr_moved.nii.gz")

        #break

    print (mr_mr_inv_results)
    print (ct_mr_inv_results)
    print (mr_mr_abs_results)
    print (ct_mr_abs_results)

    print ("Mean MR-MR registration with norm(AB^hat-I) error:", np.mean (mr_mr_inv_results))
    print ("Mean CT-MR registration with norm(AB^hat-I) error:", np.mean (ct_mr_inv_results))
    print ("Mean MR-MR registration with norm(B-B^hat) error:",  np.mean (mr_mr_abs_results))
    print ("Mean CT-MR registration with norm(B-B^hat) error:",  np.mean (ct_mr_abs_results))
