import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
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
                mi = np.load (data_folder + model_folder + "mi.npy")
                print (f"data: theta{data_rot} T{data_trans} S{data_shear} model theta{model_rot} T{model_trans} S{model_shear} {model_vers}")
                print (np.mean (mi))

    for data_rot, data_trans, data_shear in data_combinations:
        if data_shear is None:
            for model_vers in models:
                data_folder = f"aug_data/test_norm_rot{data_rot}_trans{data_trans}_shear{data_shear}/"
                model_folder = f"{model_vers}_results/"
                mi = np.load (data_folder + model_folder + "mi.npy")
                print (f"data: theta{data_rot} T{data_trans} S{data_shear} model {model_vers}")
                print (np.mean (mi))
