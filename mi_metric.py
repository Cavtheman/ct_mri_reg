import os
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from voxelmorph.tf.losses import MutualInformation, Dice

if __name__ == "__main__":
    fixed_path = sys.argv[1]
    moved_path = sys.argv[2] + "moved_ct_to_mr/ct/"
    #base_path = sys.argv[1]
    #rigid = sys.argv[2].lower() in ("true", "1", "yes", "rigid")
    #mode_str = "rigid" if rigid else "affine"

    #a = nib.load ("aug_data/test_norm_rot0.2_trans20_shearNone/affine_results/moved_ct_to_mr/ct/0.nii.gz")
    #moved_path = base_path + f"{mode_str}_results/moved_ct_to_mr/mr/"
    #fixed_path = base_path + "fixed/mr/"
    #moved_path = fixed_path

    all_moved = [ moved_path + elem for elem in os.listdir (moved_path) ]
    all_moved.sort()
    all_fixed = [ fixed_path + elem for elem in os.listdir (fixed_path) ]
    all_fixed.sort()

    #print (all_moved[:5])
    #print (all_fixed[:5])
    mi = MutualInformation ()
    dice = Dice()
    inv_loss = np.load (sys.argv[2] + "ct_mr_inv.npy")
    abs_loss = np.load (sys.argv[2] + "ct_mr_abs.npy")

    mi_vals = []
    dice_vals = []
    for i, (moved, fixed) in enumerate (zip (all_moved, all_fixed)):
        #moved = np.expand_dims (nib.load (fixed).get_fdata(), axis=(0,-1)).astype(np.float32)
        #moved = nib.load (moved).get_fdata().astype(np.float32)
        moved = np.expand_dims (nib.load (moved).get_fdata(), axis=(0,-1)).astype(np.float32)
        fixed = np.expand_dims (nib.load (fixed).get_fdata(), axis=(0,-1)).astype(np.float32)
        print (moved.shape)
        print (fixed.shape)

        #print (np.min (moved), np.max (moved))
        #print (np.min (fixed), np.max (fixed))
        print (mi.loss (fixed, moved), inv_loss[i], abs_loss[i], flush=True)
        mi_vals.append (mi.loss (fixed, moved))
        dice_vals.append (dice.loss (tf.constant (fixed), tf.constant (moved)))
        #if i > 0: break
        #a = a.get_fdata().astype(np.float32)
        #b = np.expand_dims (b.get_fdata(), axis=(0,-1)).astype(np.float32)
        #print (a.shape)
        #print (b.shape)
        #print (np.max (a))
        #print (np.max (b))
        #print (loss.loss (a,b))
        #print (np.load (base_path + "affine_results/ct_mr_inv.npy"))
        if i%10 == 0:
            print (i, flush=True)
    print (f"With fixed:", sys.argv[1])
    print (f"And moved:", sys.argv[2])
    print ("MI:", np.mean (mi_vals))
    print ("Dice:", np.mean (dice_vals))
    np.save (sys.argv[2] + "mi.npy", np.array (mi_vals))
