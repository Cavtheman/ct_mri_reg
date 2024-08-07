import os
import sys
import torch
import numpy as np
import nibabel as nib
import voxelmorph as vxm
#import tensorflow as tf

from torch.utils import data
from data_generator import AugmentData
from nibabel.orientations import axcodes2ornt, io_orientation, apply_orientation, inv_ornt_aff

def __apply_orientation__ (nib_img, orientation):
        new_or = axcodes2ornt (orientation)
        old_or = io_orientation (nib_img.affine)
        if (new_or == old_or).all(): return nib_img

        transform = nib.orientations.ornt_transform(old_or, new_or)
        data = apply_orientation (nib_img.get_fdata(), transform)
        new_affine = nib_img.affine @ inv_ornt_aff (transform, data.shape)
        ret_val = nib.Nifti1Image (data, new_affine, header=nib_img.header)
        return ret_val

def generate_images(n=None, rotate=None, translate=None, shear=None, normalise=False, pad_to=None, orientation=None):
    path = "./data/synthrad/brain/"
    aug_path = "./aug_data/"
    all_image_folders = [ path + img + "/" for img in os.listdir (path)]

    all_ct, all_mr, all_masks = map (list, (zip(*[ (img_folder + "ct.nii.gz", img_folder + "mr.nii.gz", img_folder + "mask.nii.gz")
                                                   for img_folder in all_image_folders
                                                   if "overview" not in img_folder ])))

    if n is None:
        n = len (all_ct)

    folder = f"rot{rotate}_trans{translate}_shear{shear}/"
    if normalise: folder = "norm_" + folder
    if orientation is not None: folder = f"{''.join (orientation)}_{folder}"

    # Making sure all the relevant folders have been created
    os.makedirs(f"{aug_path+folder}fixed/ct/", exist_ok=True)
    os.makedirs(f"{aug_path+folder}fixed/mr/", exist_ok=True)
    os.makedirs(f"{aug_path+folder}fixed/mask/", exist_ok=True)

    os.makedirs(f"{aug_path+folder}moving/ct/", exist_ok=True)
    os.makedirs(f"{aug_path+folder}moving/mr/", exist_ok=True)
    os.makedirs(f"{aug_path+folder}moving/mask/", exist_ok=True)

    os.makedirs(f"{aug_path+folder}transforms/", exist_ok=True)
    os.makedirs(f"{aug_path+folder}inv_transforms/", exist_ok=True)

    i = 0
    dataset = AugmentData (all_ct,
                           all_mr,
                           all_masks,
                           rotate=rotate,
                           translate=translate,
                           shear=shear,
                           pad_to=pad_to,
                           normalise=normalise,
                           orientation=orientation,
                           aug_path=aug_path+folder)

    # not using a dataloader because it requires a collate_fn function
    # which is annoying to manually implement
    #dataloader = data.DataLoader(dataset,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             num_workers=num_workers)


    # Data generation loop
    while True:
        for ct_fixed_nib, mr_fixed_nib, mask_fixed_nib, ct_moving_nib, mr_moving_nib, mask_moving_nib, transform, inv_transform in dataset:
            if i%10 == 0:
                print (i, flush=True)

            nib.save (ct_fixed_nib,
                      f"{aug_path+folder}fixed/ct/{i}.nii.gz")
            nib.save (mr_fixed_nib,
                      f"{aug_path+folder}fixed/mr/{i}.nii.gz")
            nib.save (mask_fixed_nib,
                      f"{aug_path+folder}fixed/mask/{i}.nii.gz")
            nib.save (ct_moving_nib,
                      f"{aug_path+folder}moving/ct/{i}.nii.gz")
            nib.save (mr_moving_nib,
                      f"{aug_path+folder}moving/mr/{i}.nii.gz")
            nib.save (mask_moving_nib,
                      f"{aug_path+folder}moving/mask/{i}.nii.gz")
            # Affine transform
            torch.save (transform, f"{aug_path+folder}transforms/{i}.pt")
            torch.save (inv_transform, f"{aug_path+folder}inv_transforms/{i}.pt")

            i += 1
            if i >= n: break
        if i >= n: break


if __name__ == "__main__":
    num_workers = 0

    n = 720

    # Arguments varied at runtime
    rotate = None if sys.argv[1] == "0" else float(sys.argv[1])
    translate = None if sys.argv[2] == "0" else int(sys.argv[2])
    shear = None if sys.argv[3] == "0" else float(sys.argv[3])
    normalise = sys.argv[4].lower() in ("true", "1", "yes")
    orientation = tuple (sys.argv[5])

    # not necessary since orientation is done here. But kept just in case
    if orientation == ("L","I","A"):
        pad_to = (280,262,284)
    elif orientation == ("R","A","S"):
        pad_to = (284,280,262)
    else:
        pad_to = (280,284,262)
    #(280,284,262) base synthrad orientation
    #(280,262,284), LIA orientation
    #(284,280,262), RAS orientation
    # Or just None


    #pad_to = (280,262,284)
    #pad_to = (280,284,262)

    generate_images (n=n,
                     rotate=rotate,
                     translate=translate,
                     shear=shear,
                     normalise=normalise,
                     pad_to=pad_to,
                     orientation=orientation)

    #generate_images (n=n, rotate=rotate, translate=translate, shear=0.1, normalise=True)
    #generate_images (n=n, rotate=rotate, translate=translate, shear=None, normalise=True)
    #generate_images (n=n, rotate=rotate, translate=translate, shear=0.1, normalise=False)
    #generate_images (n=n, rotate=rotate, translate=translate, shear=None, normalise=False)
