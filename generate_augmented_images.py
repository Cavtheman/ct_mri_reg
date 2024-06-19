import os
import torch
import numpy as np
import nibabel as nib
#import tensorflow as tf
from torch.utils import data
from data_generator import AugmentData

print ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

def generate_images(n, rotate=None, translate=None, shear=None, normalise=False):
    path = "./data/synthrad/brain/"
    aug_path = "./aug_data/"
    all_image_folders = [ path + img + '/' for img in os.listdir (path)]

    all_ct, all_mr, all_masks = map (list, (zip(*[ (img_folder + "ct.nii.gz", img_folder + "mr.nii.gz", img_folder + "mask.nii.gz")
                                                   for img_folder in all_image_folders
                                                   if "overview" not in img_folder ])))

    folder = f"rot{rotate}_trans{translate}_shear{shear}/"
    if normalise: folder = "norm_" + folder

    if not os.path.exists(aug_path + folder): os.makedirs(aug_path + folder)

    i = 0
    # No image is larger than (280, 284, 262)
    dataset = AugmentData (all_ct,
                           all_mr,
                           all_masks,
                           rotate=rotate,
                           translate=translate,
                           shear=shear,
                           pad_to=pad_val,
                           normalise=normalise)
    dataloader = data.DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=num_workers)


    while True:
        for fixed, fixed_mask, moving, aug_moving, aug_mask, transform, inv_transform in dataloader:
            if i%50 == 0:
                print (i, flush=True)
            # Saving like this won't work with batches larger than 1
            # Note that this has thrown away the original world coordinates
            nib.save (nib.Nifti1Image (fixed.squeeze().numpy(), np.eye(4)),
                      aug_path + folder + f"{i}_fixed_ct.nii.gz")

            nib.save (nib.Nifti1Image (fixed_mask.squeeze().numpy(), np.eye(4)),
                      aug_path + folder + f"{i}_fixed_mask.nii.gz")

            nib.save (nib.Nifti1Image (moving.squeeze().numpy(), np.eye(4)),
                      aug_path + folder + f"{i}_orig_mr.nii.gz")

            # Note that the augmented image has the affine transform to get it back to the original saved in .affine
            nib.save (nib.Nifti1Image (aug_moving.squeeze().numpy(), inv_transform.squeeze()),
                      aug_path + folder + f"{i}_aug_mr.nii.gz")

            nib.save (nib.Nifti1Image (aug_mask.squeeze().numpy(), inv_transform.squeeze()),
                      aug_path + folder + f"{i}_aug_mask.nii.gz")

            # Affine transform
            torch.save (transform.squeeze(), aug_path + folder + f"{i}_transform.pt")

            i += 1
            if i >= n: break
        if i >= n: break


if __name__ == "__main__":
    num_workers = 0
    pad_val = (280, 284, 262)

    n = 1
    rotate = 0.2
    translate = 20
    shear = None
    normalise = True
    #generate_images (n=n, rotate=rotate, translate=translate, normalise=normalise)
    generate_images (n=n, rotate=rotate, translate=translate, shear=shear, normalise=normalise)
