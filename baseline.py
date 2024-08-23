import os
import re
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
    #norm_fixed = fixed
    #norm_moving = moving

    if fixed_mask is not None:
        fixed = fixed * fixed_mask
    if moving_mask is not None:
        moving = moving * moving_mask

    pred_transform = model.predict ((moving, fixed), verbose=0)

    if transform:
        # make_dense=False in the affine model means shift_center must be False as well
        out_img = vxm.layers.SpatialTransformer(interp_method="nearest",fill_value=0,shift_center=False)((moving, pred_transform))

    # synthmorph output just doesn't include the bottom line of the affine matrix for some reason
    pred_transform = np.concatenate ((pred_transform, [[[0,0,0,1]]]), axis=1)

    if transform:
        return out_img.numpy().astype(np.float32).squeeze(), pred_transform
    else:
        return pred_transform


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    num_workers = 0
    n = 720
    output_transformed = True

    aug_path = sys.argv[1]
    rigid = sys.argv[2].lower() in ("rigid", "true", "1", "yes")
    loaded_model = sys.argv[3] if len (sys.argv) == 4 else None
    #aug_path = "./aug_data/side_norm_rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/norm_rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/norm_rot0.2_trans20_shear0.1/"
    #aug_path = "./aug_data/rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/rot0.2_trans20_shear0.1/"

    mode_str = "rigid" if rigid else "affine"
    if loaded_model is not None:
        pattern = r"rot(?P<rot>[0-9.]+)_trans(?P<trans>[0-9.]+)_shear(?P<shear>[0-9.]+|None)/finetune_(?P<type>rigid|affine)"

        # Search for the pattern in the path
        match = re.search(pattern, loaded_model)

        if match:
            # Extracted values
            rot = float (match.group("rot"))
            trans = int (match.group("trans"))
            shear = None if match.group("shear") == "None" else float (match.group("shear"))
            model_type = match.group ("type")

        save_folder = f"{aug_path}/rot{rot}_trans{trans}_shear{shear}_{model_type}_model_results/"
    else:
        save_folder = f"{aug_path}/{mode_str}_results/"

    # Making sure all the relevant folders have been created
    os.makedirs(f"{save_folder}moved_ct_to_mr/ct/", exist_ok=True)
    os.makedirs(f"{save_folder}moved_ct_to_mr/mr/", exist_ok=True)
    os.makedirs(f"{save_folder}moved_mr_to_mr/", exist_ok=True)
    os.makedirs(f"{save_folder}pred_transforms/mr_mr/", exist_ok=True)
    os.makedirs(f"{save_folder}pred_transforms/ct_mr/", exist_ok=True)


    aug_data = SynthradData (aug_path,
                             include_mask=False)
    #dataloader = data.DataLoader(aug_data,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             num_workers=num_workers)

    # Registration model.
    # Shape is hardcoded for the sake of speed, but found from AugmentData.__max_size__(). Corresponds to maximal shape of all images
    #in_shape = (280, 284, 262)
    #in_shape = (280, 262, 284) # transposed for LIA
    in_shape = aug_data.shape ()

    model_aff = vxm.tf.networks.VxmAffineFeatureDetector(in_shape, rigid=rigid, make_dense=False)

    if loaded_model is not None:
        model_aff.load_weights (loaded_model)
    elif rigid:
        #model_aff.load_weights ("synthmorph.rigid.1.h5")
        model_aff.load_weights ("./freesurfer/models/synthmorph_rigid.h5")
    else:
        model_aff.load_weights ("./freesurfer/models/synthmorph_affine.h5")
        #model_aff.load_weights ("synthmorph.affine.crop.h5")

    ct_mr_inv_results = np.zeros (len (aug_data))
    mr_mr_inv_results = np.zeros (len (aug_data))

    ct_mr_abs_results = np.zeros (len (aug_data))
    mr_mr_abs_results = np.zeros (len (aug_data))

    for i, (ct_fixed_nib, mr_fixed_nib, ct_moving_nib, mr_moving_nib, transform, inv_transform) in enumerate (aug_data):
        ct_fixed = ct_fixed_nib.get_fdata()
        mr_fixed = mr_fixed_nib.get_fdata()
        ct_moving = ct_moving_nib.get_fdata()
        mr_moving = mr_moving_nib.get_fdata()

        ct_fixed = np.expand_dims (ct_fixed, (0,-1))
        mr_fixed = np.expand_dims (mr_fixed, (0,-1))
        ct_moving = np.expand_dims (ct_moving, (0,-1))
        mr_moving = np.expand_dims (mr_moving, (0,-1))

        if output_transformed:
            ct_to_mr_result_ct, pred_transform_1 = register (model_aff, mr_fixed, ct_moving, transform=output_transformed)
            mr_to_mr_result, pred_transform_2 = register (model_aff, mr_fixed, mr_moving, transform=output_transformed)

            # Using the ct-mr transform to move an mr, for use with the segmentation-based evaluation
            ct_to_mr_result_mr = vxm.layers.SpatialTransformer(interp_method="nearest",fill_value=0,shift_center=False)((mr_moving, (pred_transform_1[:,:3]))).numpy().astype(np.float32).squeeze()
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
            ct_to_mr_result_ct_affine = ct_moving_nib.affine @ pred_transform_1.squeeze()
            ct_to_mr_result_ct = nib.Nifti1Image (ct_to_mr_result_ct.squeeze(),
                                                  affine=ct_to_mr_result_ct_affine,
                                                  header=ct_moving_nib.header)
            ct_to_mr_result_mr_affine = mr_moving_nib.affine @ pred_transform_1.squeeze()
            ct_to_mr_result_mr = nib.Nifti1Image (ct_to_mr_result_mr.squeeze(),
                                                  affine=ct_to_mr_result_mr_affine,
                                                  header=mr_moving_nib.header)

            mr_to_mr_result_affine = mr_moving_nib.affine @ pred_transform_2.squeeze()
            mr_to_mr_result = nib.Nifti1Image (mr_to_mr_result.squeeze(),
                                               affine=mr_to_mr_result_affine,
                                               header=mr_moving_nib.header)
            nib.save (ct_to_mr_result_ct,
                      f"{save_folder}moved_ct_to_mr/ct/{i}.nii.gz")
            nib.save (ct_to_mr_result_mr,
                      f"{save_folder}moved_ct_to_mr/mr/{i}.nii.gz")
            nib.save (mr_to_mr_result,
                      f"{save_folder}moved_mr_to_mr/{i}.nii.gz")

            torch.save (torch.tensor (pred_transform_1),
                        f"{save_folder}pred_transforms/mr_mr/{i}.pt")
            torch.save (torch.tensor (pred_transform_2),
                        f"{save_folder}pred_transforms/ct_mr/{i}.pt")

        if i%10 == 0:
            print (i, flush=True)
        if i >= n-1: break

    print (mr_mr_inv_results[:n])
    print (ct_mr_inv_results[:n])
    print (mr_mr_abs_results[:n])
    print (ct_mr_abs_results[:n])

    np.save (f"{save_folder}mr_mr_inv.npy", mr_mr_inv_results[:n])
    np.save (f"{save_folder}ct_mr_inv.npy", ct_mr_inv_results[:n])
    np.save (f"{save_folder}mr_mr_abs.npy", mr_mr_abs_results[:n])
    np.save (f"{save_folder}ct_mr_abs.npy", ct_mr_abs_results[:n])

    print ("Mean MR-MR registration with norm(AB^hat-I) error:", np.mean (mr_mr_inv_results[:n]))
    print ("Mean CT-MR registration with norm(AB^hat-I) error:", np.mean (ct_mr_inv_results[:n]))
    print ("Mean MR-MR registration with norm(B-B^hat) error:",  np.mean (mr_mr_abs_results[:n]))
    print ("Mean CT-MR registration with norm(B-B^hat) error:",  np.mean (ct_mr_abs_results[:n]))
