import os
import sys
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
    norm_fixed = fixed.numpy()
    norm_moving = moving.numpy()

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

    num_workers = 0
    n = 180
    output_transformed = True
    rigid = True

    aug_path = sys.argv[1]
    #aug_path = "./aug_data/side_norm_rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/norm_rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/norm_rot0.2_trans20_shear0.1/"
    #aug_path = "./aug_data/rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/rot0.2_trans20_shear0.1/"

    if not os.path.exists(aug_path + "moved_ct_to_mr/ct/"): os.makedirs(aug_path + "moved_ct_to_mr/ct/")
    if not os.path.exists(aug_path + "moved_ct_to_mr/mr/"): os.makedirs(aug_path + "moved_ct_to_mr/mr/")
    if not os.path.exists(aug_path + "moved_mr_to_mr/"): os.makedirs(aug_path + "moved_mr_to_mr/")

    # Registration model.
    # Shape is hardcoded for the sake of speed, but found from AugmentData.__max_size__(). Corresponds to maximal shape of all images
    in_shape = (280, 284, 262)
    model_aff = vxm.tf.networks.VxmAffineFeatureDetector(in_shape, rigid=rigid, make_dense=False)

    if rigid:
        #model_aff.load_weights ("synthmorph.rigid.1.h5")
        model_aff.load_weights ("./freesurfer/models/synthmorph_rigid.h5")
    else:
        model_aff.load_weights ("./freesurfer/models/synthmorph_affine.h5")
        #model_aff.load_weights ("synthmorph.affine.crop.h5")

    aug_data = SynthradData (aug_path,
                             include_mask=False)
    dataloader = data.DataLoader(aug_data,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=num_workers)

    ct_mr_inv_results = np.zeros (len (aug_data))
    mr_mr_inv_results = np.zeros (len (aug_data))

    ct_mr_abs_results = np.zeros (len (aug_data))
    mr_mr_abs_results = np.zeros (len (aug_data))

    for i, (ct_fixed, mr_fixed, ct_moving, mr_moving, transform, inv_transform) in enumerate (dataloader):
        if output_transformed:
            ct_to_mr_result_ct, pred_transform_1 = register (model_aff, mr_fixed, ct_moving, transform=output_transformed)
            mr_to_mr_result, pred_transform_2 = register (model_aff, mr_fixed, mr_moving, transform=output_transformed)

            # Using the ct-mr transform to move an mr, for use with the segmantation-based evaluation
            ct_to_mr_result_mr = vxm.layers.SpatialTransformer(fill_value=0,shift_center=False)((mr_moving.numpy(), (pred_transform_1[:,:3]))).numpy().astype(np.float64)
        else:
            pred_transform_1 = register (model_aff, mr_fixed, ct_moving, transform=output_transformed)
            pred_transform_2 = register (model_aff, mr_fixed, mr_moving, transform=output_transformed)

        # First error is calculated based on knowledge that the predicted transform is the inverse
        # of the one used to generate it. So transform x pred_transform should become an identity
        # matrix. The 2-norm of this is then taken to calculate error value
        ct_mr_inv_results[i] = norm (np.matmul (transform, pred_transform_1) - np.eye(4))
        mr_mr_inv_results[i] = norm (np.matmul (transform, pred_transform_2) - np.eye(4))

        # Since the inverse transformation matrix is available, I use that as well
        ct_mr_abs_results[i] = norm (inv_transform - pred_transform_1)
        mr_mr_abs_results[i] = norm (inv_transform - pred_transform_2)


        # Saving an image pair for visualisation
        if output_transformed:
            ct_to_mr_result_ct = nib.Nifti1Image (ct_to_mr_result_ct, affine=pred_transform_1.squeeze())
            ct_to_mr_result_mr = nib.Nifti1Image (ct_to_mr_result_mr, affine=pred_transform_1.squeeze())

            mr_to_mr_result = nib.Nifti1Image (mr_to_mr_result, affine=pred_transform_2.squeeze())
            nib.save (ct_to_mr_result_ct, aug_path + "moved_ct_to_mr/ct/" + f"{i}.nii.gz")
            nib.save (ct_to_mr_result_mr, aug_path + "moved_ct_to_mr/mr/" + f"{i}.nii.gz")

            nib.save (mr_to_mr_result, aug_path + "moved_mr_to_mr/" + f"{i}.nii.gz")

        if i%50 == 0:
            print (i, flush=True)
        if i >= n-1: break

    print (mr_mr_inv_results[:n])
    print (ct_mr_inv_results[:n])
    print (mr_mr_abs_results[:n])
    print (ct_mr_abs_results[:n])

    np.save (aug_path + "mr_mr_inv_results.npy", mr_mr_inv_results[:n])
    np.save (aug_path + "ct_mr_inv_results.npy", ct_mr_inv_results[:n])
    np.save (aug_path + "mr_mr_abs_results.npy", mr_mr_abs_results[:n])
    np.save (aug_path + "ct_mr_abs_results.npy", ct_mr_abs_results[:n])

    print ("Mean MR-MR registration with norm(AB^hat-I) error:", np.mean (mr_mr_inv_results[:n]))
    print ("Mean CT-MR registration with norm(AB^hat-I) error:", np.mean (ct_mr_inv_results[:n]))
    print ("Mean MR-MR registration with norm(B-B^hat) error:",  np.mean (mr_mr_abs_results[:n]))
    print ("Mean CT-MR registration with norm(B-B^hat) error:",  np.mean (ct_mr_abs_results[:n]))
