import os
import sys
import torch
import numpy as np
import nibabel as nib
#import tensorflow as tf
from torch.utils import data
from data_generator import AugmentData
import voxelmorph as vxm

def generate_images(n, rotate=None, translate=None, shear=None, normalise=False, side_view=False):
    path = "./data/synthrad/brain/"
    aug_path = "./aug_data/"
    all_image_folders = [ path + img + '/' for img in os.listdir (path)]

    all_ct, all_mr, all_masks = map (list, (zip(*[ (img_folder + "ct.nii.gz", img_folder + "mr.nii.gz", img_folder + "mask.nii.gz")
                                                   for img_folder in all_image_folders
                                                   if "overview" not in img_folder ])))

    folder = f"rot{rotate}_trans{translate}_shear{shear}/"
    if normalise: folder = "norm_" + folder
    if side_view: folder = "side_" + folder

    if not os.path.exists(aug_path + folder + "fixed/ct/"): os.makedirs(aug_path + folder + "fixed/ct/")
    if not os.path.exists(aug_path + folder + "fixed/mr/"): os.makedirs(aug_path + folder + "fixed/mr/")
    if not os.path.exists(aug_path + folder + "fixed/mask/"): os.makedirs(aug_path + folder + "fixed/mask/")

    if not os.path.exists(aug_path + folder + "moving/ct/"): os.makedirs(aug_path + folder + "moving/ct/")
    if not os.path.exists(aug_path + folder + "moving/mr/"): os.makedirs(aug_path + folder + "moving/mr/")
    if not os.path.exists(aug_path + folder + "moving/mask/"): os.makedirs(aug_path + folder + "moving/mask/")

    if not os.path.exists(aug_path + folder + "transforms/"): os.makedirs(aug_path + folder + "transforms/")

    i = 0
    # No image is larger than (280, 284, 262)
    dataset = AugmentData (all_ct,
                           all_mr,
                           all_masks,
                           rotate=rotate,
                           translate=translate,
                           shear=shear,
                           pad_to=pad_val,
                           normalise=normalise,
                           side_view=side_view)
    dataloader = data.DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=num_workers)


    while True:
        for ct_fixed, mr_fixed, mask_fixed, ct_moving, mr_moving, mask_moving, transform, inv_transform in dataloader:
            if i%50 == 0:
                print (i, flush=True)
            # Saving like this won't work with batches larger than 1
            # Note that this has thrown away the original world coordinates
            nib.save (nib.Nifti1Image (ct_fixed.squeeze().numpy(), np.eye(4)),
                      aug_path + folder + "fixed/ct/" + f"{i}.nii.gz")

            nib.save (nib.Nifti1Image (mr_fixed.squeeze().numpy(), np.eye(4)),
                      aug_path + folder + "fixed/mr/" + f"{i}.nii.gz")

            nib.save (nib.Nifti1Image (mask_fixed.squeeze().numpy(), np.eye(4)),
                      aug_path + folder + "fixed/mask/" + f"{i}.nii.gz")


            # Note that the augmented image has the affine transform to get it back to the original saved in .affine
            nib.save (nib.Nifti1Image (ct_moving.squeeze().numpy(), inv_transform.squeeze()),
                      aug_path + folder + "moving/ct/" + f"{i}.nii.gz")

            nib.save (nib.Nifti1Image (mr_moving.squeeze().numpy(), inv_transform.squeeze()),
                      aug_path + folder + "moving/mr/" + f"{i}.nii.gz")

            nib.save (nib.Nifti1Image (mask_moving.squeeze().numpy(), inv_transform.squeeze()),
                      aug_path + folder + "moving/mask/" + f"{i}.nii.gz")

            # Affine transform
            torch.save (transform.squeeze(), aug_path + folder + "transforms/" + f"{i}.pt")

            i += 1
            if i >= n: break
        if i >= n: break


if __name__ == "__main__":
    num_workers = 0
    pad_val = (280, 284, 262)


    n = 180
    side_view = False

    # Arguments varied at runtime
    rotate = None if sys.argv[1] == "0" else float(sys.argv[1])
    translate = None if sys.argv[2] == "0" else int(sys.argv[2])
    shear = None if sys.argv[3] == "0" else float(sys.argv[3])
    normalise = sys.argv[4].lower() in ('true', '1', 'yes')

    if side_view:
        pad_val = np.transpose(pad_val, [2,1,0])
    generate_images (n=n,
                     rotate=rotate,
                     translate=translate,
                     shear=shear,
                     normalise=normalise,
                     side_view=side_view)

    #generate_images (n=n, rotate=rotate, translate=translate, shear=0.1, normalise=True)
    #generate_images (n=n, rotate=rotate, translate=translate, shear=None, normalise=True)
    #generate_images (n=n, rotate=rotate, translate=translate, shear=0.1, normalise=False)
    #generate_images (n=n, rotate=rotate, translate=translate, shear=None, normalise=False)
