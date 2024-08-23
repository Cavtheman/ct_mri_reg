import os
import numpy as np
import einops as ein
import nibabel as nib

from torch.utils import data
from data_generator import SynthradData
from utils import plot_volume, normalise_array

def find_files (path):
        ret_val = [ path + img for img in os.listdir (path) ]
        ret_val.sort()
        return ret_val

if __name__ == "__main__":
    num_workers = 0

    img_num = 0

    data_rot = 0.4
    data_trans = 40
    data_shear = None

    model_rot = 0.4
    model_trans = 40
    model_shear = None
    model_vers = "rigid"

    orientation = (1,2,0) # side
    #orientation = (0,1,2)
    #"./aug_data/test_norm_rot0.4_trans40_shearNone/fixed/mr/"
    # "./aug_data/test_norm_rot0.4_trans40_shearNone/rot0.2_trans20_shearNone_rigid_model_results/"
    data_folder = f"aug_data/test_norm_rot{data_rot}_trans{data_trans}_shear{data_shear}/"
    model_folder = f"rot{model_rot}_trans{model_trans}_shear{model_shear}_{model_vers}_model_results/"

    fixed_mr = nib.load (find_files (data_folder + "fixed/mr/")[img_num])
    fixed_mr = np.transpose (fixed_mr.get_fdata(), orientation)
    fixed_ct = nib.load (find_files (data_folder + "fixed/ct/")[img_num])
    fixed_ct = np.transpose (fixed_ct.get_fdata(), orientation)

    moving_mr = nib.load (find_files (data_folder + "moving/mr/")[img_num])
    moving_mr = np.transpose (moving_mr.get_fdata(), orientation)
    moving_ct = nib.load (find_files (data_folder + "moving/ct/")[img_num])
    moving_ct = np.transpose (moving_ct.get_fdata(), orientation)

    moved_mr = nib.load (find_files (data_folder + model_folder + "moved_mr_to_mr/")[img_num])
    #moved_mr = nib.load (find_files (data_folder + model_folder + "moved_ct_to_mr/mr/")[img_num])
    moved_mr = np.transpose (moved_mr.get_fdata().squeeze(), orientation)
    moved_ct = nib.load (find_files (data_folder + model_folder + "moved_ct_to_mr/ct/")[img_num])
    moved_ct = np.transpose (moved_ct.get_fdata().squeeze(), orientation)

    gt_segmented = nib.load (find_files (data_folder + model_folder + "segmentations/fixed/")[img_num])
    gt_segmented = np.transpose (gt_segmented.get_fdata().squeeze(), orientation)

    segmented = nib.load (find_files (data_folder + model_folder + "segmentations/moved/")[img_num])
    segmented = np.transpose (segmented.get_fdata().squeeze(), orientation)

    inv_loss = np.load (data_folder + model_folder + "ct_mr_inv.npy")
    abs_loss = np.load (data_folder + model_folder + "ct_mr_abs.npy")

    print (np.mean (inv_loss))
    print (np.mean (abs_loss))

    print (inv_loss[img_num])
    print (abs_loss[img_num])


    #plot_volume (fixed_mr - moved_mr)
    #plot_volume (fixed_ct - moved_ct)
    square = np.concatenate ((np.concatenate ([fixed_mr, moving_mr], axis=1),
                              np.concatenate ([fixed_ct, moving_ct], axis=1)), axis=0)
    #plot_volume (square)
    #plot_volume (np.concatenate ([moving_mr, fixed_mr, moved_mr, moved_ct], axis=1))
    #plot_volume (np.concatenate ([moving_mr, fixed_mr, moved_mr], axis=1))
    #plot_volume (np.concatenate ([moving_ct, fixed_ct, moved_ct], axis=1))
    #plot_volume (fixed_mr-moved_mr)

    #plot_volume (fixed_mr)
    #plot_volume (moving_mr)
    #plot_volume (moved_mr)

    #plot_volume (np.stack ([fixed_mr, moved_mr, moving_mr], axis=3),
    #             rgb=True,
    #             labels=("Fixed", "Moved using CT", "Moving"))
    axes=np.array((0,2,1))
    plot_volume ((np.concatenate ([np.transpose (gt_segmented, axes=axes),
                                   np.transpose (segmented, axes=axes)],
                                  axis=1)),
                 #rgb=True,
                 cmap="tab20")

    #plot_volume (np.stack ([fixed_ct, moved_ct, moving_ct], axis=3),
    #             rgb=True,
    #             labels=("Fixed", "Moved using CT", "Moving"))

    #aug_path = "./aug_data/test_norm_rot0.4_trans40_shearNone/"
    #aug_path = "./aug_data/test_norm_rot0.4_trans40_shear0.1/"
    #aug_path = "./aug_data/test_norm_rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/test_norm_rot0.2_trans20_shearNone/"

    #model_folder = "affine_results/"
    #model_folder = "rigid_results/"
    #model_folder = "rot0.2_trans20_shearNone_rigid_model_results/"
    #aug_data/test_norm_rot0.2_trans20_shearNone

    #ct_mr = nib.load ("ct_mr_moved.nii.gz")
    #mr_mr = nib.load ("mr_mr_moved.nii.gz")

    #ct_mr = nib.load (aug_path + model_folder + "moved_ct_to_mr/mr/1.nii.gz")
    #mr_mr = nib.load (aug_path + model_folder + "moved_mr_to_mr/1.nii.gz")


    #ct_mr = nib.load (aug_path + "rigid_results/moved_ct_to_mr/ct/0.nii.gz")
    #mr_mr = nib.load (aug_path + "rigid_results/moved_mr_to_mr/0.nii.gz")

    #segmented = nib.load (aug_path + model_folder + "segmentations/moved/0_synthseg.nii.gz").get_fdata()

    #ct_mr = ct_mr.get_fdata().squeeze()
    #mr_mr = mr_mr.get_fdata().squeeze()

    #print (ct_mr.shape)
    #print (mr_mr.shape)


    #aug_data = SynthradData (aug_path)
    #dataloader = data.DataLoader(aug_data,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             num_workers=num_workers)

#    for ct_fixed, mr_fixed, mask_fixed, ct_moving, mr_moving, mask_moving, target_transform, aug_transform in dataloader:
#        #plot_volume (ct_mr)
#        print (mr_fixed.shape)
#        print (mr_fixed.squeeze().shape)
#        print (mr_moving.shape)
#        #plot_volume (mr_moving.squeeze().numpy() + mr_mr + mr_fixed.squeeze().numpy())
#        print (np.min (mr_fixed.numpy()))
#        print (np.max (mr_fixed.numpy()))
#
#        print (np.min (mr_moving.numpy()))
#        print (np.max (mr_moving.numpy()))
#
#        print (np.min (ct_fixed.numpy()))
#        print (np.max (ct_fixed.numpy()))
#
#        print (np.min (mr_mr))
#        print (np.max (mr_mr))
#
#        mr_fixed = normalise_array (mr_fixed.squeeze().numpy())
#        mr_moving = normalise_array (mr_moving.squeeze().numpy())
#        ct_moving = normalise_array (ct_moving.squeeze().numpy())
#        ct_fixed = normalise_array (ct_fixed.squeeze().numpy())
#        mr_mr = normalise_array (mr_mr)
#        ct_mr = normalise_array (ct_mr)
#
#        #print (np.unique (segmented.astype(int), return_counts=True))
#        #plot_volume (segmented.astype(int), rgb=True, cmap="tab10")
#        #plot_volume (np.stack ([mr_mr, mr_mr, segmented.astype(int)], axis=3), rgb=True, cmap="tab20")
#        #test = np.stack ([mr_moving, ct_fixed, mr_mr], axis=3)
#        #plot_volume (ct_mr)
#        plot_volume (np.stack ([mr_fixed, ct_mr, mr_moving], axis=3),
#                     side_view=True,
#                     rgb=True,
#                     labels=("Fixed", "Moved using MR", "Moving"))

        #plot_volume (np.stack ([mr_mr], axis=3),
        #             side_view=False,
        #             rgb=True,
        #             labels=("Fixed", "Moved using MR", "Moving"))

        #plot_volume (np.stack ([mr_fixed, ct_mr, mr_moving], axis=3),
        #             side_view=True,
        #             rgb=True,
        #             labels=("Fixed", "Moved using CT", "Moving"))

        #plot_volume (np.stack ([ct_fixed * mask_fixed.squeeze().numpy(), mr_moving * mask_moving.squeeze().numpy(), orig_moving * mask_fixed.squeeze().numpy()], axis=3), rgb=True)
        #plot_volume (moving - moving * moving_mask.squeeze().numpy())
        #break
