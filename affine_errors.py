import os
import sys
import math
import torch
import numpy as np

def get_rotation(affine, first_solution=True):
    R = affine[:3, :3]

    pitch = -math.asin(R[2, 0])
    roll = math.atan2(R[2, 1] / math.cos(pitch), R[2, 2] / math.cos(pitch))
    yaw = math.atan2(R[1, 0] / math.cos(pitch), R[0, 0] / math.cos(pitch))

    pitch2 = math.pi - pitch
    roll2 = math.atan2(R[2, 1] / math.cos(pitch2), R[2, 2] / math.cos(pitch2))
    yaw2 = math.atan2(R[1, 0] / math.cos(pitch2), R[0, 0] / math.cos(pitch2))

    if first_solution:
        yaw_deg = math.degrees(yaw)
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)
        return yaw_deg, pitch_deg, roll_deg
    else:
        yaw_deg = math.degrees(yaw2)
        pitch_deg = math.degrees(pitch2)
        roll_deg = math.degrees(roll2)
        return yaw_deg, pitch_deg, roll_deg

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    #base_path = sys.argv[1]
    #rigid = sys.argv[2].lower() in ("true", "1", "yes", "rigid")
    #mode_str = "rigid" if rigid else "affine"
    rot_trans = [(0.2,20),(0.4,40)]
    shear_vals = [None,0.1]
    models = ["rigid","affine"]
    all_combinations = [ (rt[0], rt[1], shear, model)
                         for rt in rot_trans
                         for shear in shear_vals
                         for model in models ]
    data_combinations = [ (rt[0], rt[1], shear)
                          for rt in rot_trans
                          for shear in shear_vals ]

    for data_rot, data_trans, data_shear in data_combinations:
        if data_shear is None:
            for model_rot, model_trans, model_shear, model_vers in all_combinations:
                data_folder = f"aug_data/test_norm_rot{data_rot}_trans{data_trans}_shear{data_shear}/"
                model_folder = f"rot{model_rot}_trans{model_trans}_shear{model_shear}_{model_vers}_model_results/"
                print (data_folder)
                print (model_folder)
                target_path = data_folder + "inv_transforms/"
                pred_path = data_folder + model_folder +  "pred_transforms/ct_mr/"

                targets = os.listdir (target_path)
                preds = os.listdir (pred_path)
                targets.sort()
                preds.sort()
                print (targets)
                print (preds)
                targets = torch.stack ([ torch.load (target_path + elem) for elem in targets ])
                preds = torch.stack ([ torch.load (pred_path + elem).squeeze() for elem in preds ])

                #print (targets.size())
                #print (preds.size())
                print (torch.linalg.norm (targets - preds, dim=(1,2)))
                print (torch.linalg.norm (targets @ preds, dim=(1,2)))
                mean = torch.mean (targets - preds, axis=0)
                std = torch.std (targets - preds, axis=0)
                #print (mean.size())
                #print (std.size())
                print (f"data: theta\pm{data_rot} T\pm{data_trans} S\pm{data_shear}, model: theta\pm{model_rot} T\pm{model_trans} S\pm{model_shear} {model_vers}")
                print (mean)
                print (std)
                print ()
                break
        break
#    mean_out = torch.zeros (len (targets), 4, 4)
#    std_out = torch.zeros (len (targets), 4, 4)
#    for i, (target, pred) in enumerate (zip (targets, preds)):
#        target = torch.load (target_path + target)
#        pred = torch.load (pred_path + pred)
#        mean_out[i] = target - pred
#        std_out[i] = abs (target - pred)
#        #print (target)
#        #print (pred)
#        #print (abs (target - pred))
#
#
#    print (mean_out.size())
#    #std_out = torch.std (torch.stack ([mean_out, mean_out + 0.01], axis=0), dim=0)
#    std_out = torch.std (mean_out, dim=0)
#    #std_out[:3,:3] = torch.rad2deg (std_out[:3,:3])
#    mean_out = torch.mean (mean_out, dim=0)
#
#    print (std_out)
#    print (mean_out)
#
#    print (get_rotation (mean_out + np.eye (4)))
