import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from collections import defaultdict

def combine_by_key (dice_dict, combine_by=0):
    sums = {}
    counts = {}
    for keys, (val1, val2) in dice_dict.items():
        comb_key = keys[combine_by]
        if comb_key not in sums:
            sums[comb_key] = (val1, val2)
            counts[comb_key] = 1
        else:
            sums[comb_key] = (sums[comb_key][0] + val1, sums[comb_key][1] + val2)
            counts[comb_key] += 1
    return {key: (sums[key][0] / counts[key], sums[key][1] / counts[key]) for key in sums}


if __name__ == "__main__":
    used_labels = np.load ("SynthSeg/data/labels_classes_priors/synthseg_segmentation_labels.npy")
    used_labels, unique_idx = np.unique (used_labels, return_index=True)
    segmentation_names = np.load ("SynthSeg/data/labels_classes_priors/synthseg_segmentation_names.npy")

    segmentation_labels = segmentation_names[unique_idx]

    white_matter = [ i for i, label in enumerate (segmentation_labels) if "white matter" in label ]

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
    k = 0
    all_scores = {}
    all_mean_scores = {}
    for i, (data_rot, data_trans, data_shear) in enumerate (data_combinations):
        if data_shear is None:
            for j, (model_rot, model_trans, model_shear, model_vers) in enumerate (all_combinations):
                data_folder = f"aug_data/test_norm_rot{data_rot}_trans{data_trans}_shear{data_shear}/"
                model_folder = f"rot{model_rot}_trans{model_trans}_shear{model_shear}_{model_vers}_model_results/"
                tag = ((data_rot, data_trans, data_shear), (model_rot, model_trans, model_shear, model_vers))

                dice = np.load (data_folder + model_folder + "segmentations/moved/dice.npy")
                haus = np.load (data_folder + model_folder + "segmentations/moved/hausdorff.npy")
                mean_dice = np.mean (dice, axis=1)
                mean_haus = np.mean (haus, axis=1)
                full_mean_dice = np.mean (mean_dice)
                full_mean_haus = np.mean (mean_haus)
                all_scores[(i,j)] = (mean_dice, mean_haus)
                all_mean_scores[(i,j)] = (full_mean_dice, full_mean_haus)
                #all_mean_scores.append ((i, j, full_mean_dice, full_mean_haus))
                data = [ (label, d, h)
                         for label, d, h
                         in list (zip (segmentation_labels, dice, haus))]
                         #if label != "background"]
                labels, dice, haus = zip (*data)

                #print (dice.shape, mean_dice.shape)
                #print (mean_dice)
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))

                # Violin plot for Dice Coefficient
                dice_violins = axes[0].violinplot(dice, showmeans=True, vert=False)
                axes[0].set_yticks(np.arange(1, len(labels) + 1))
                axes[0].set_yticklabels(labels)
                axes[0].set_xlim(-0.025, 1.025)
                axes[0].set_xlabel("Dice Coefficient")
                axes[0].set_title("Dice Coefficient Distribution")

                # Violin plot for Hausdorff Distance
                haus_violins = axes[1].violinplot(haus, showmeans=True, vert=False)
                axes[1].set_yticks(np.arange(1, len(labels) + 1))
                axes[1].set_yticklabels([])
                axes[1].set_xlabel("Hausdorff Distance")
                axes[1].set_title("Hausdorff Distance Distribution")

                cmap = cm.tab20
                for partname in ("cbars","cmins","cmaxes","cmeans",):
                    violin1 = dice_violins[partname]
                    violin1.set_edgecolor("red")
                    violin1.set_linewidth(1)
                    violin2 = haus_violins[partname]
                    violin2.set_edgecolor("red")
                    violin2.set_linewidth(1)

                #for i, (violin1, violin2) in enumerate(zip (dice_violins["bodies"], haus_violins["bodies"])):
                #    color = cmap(i / len(data))
                #    violin1.set_facecolor(color)
                #    #violin1.set_edgecolor("black")
                #    violin1.set_alpha(0.7)
                #    violin2.set_facecolor(color)
                #    #violin2.set_edgecolor("black")
                #    violin2.set_alpha(0.7)

                title_model = f"$\Theta\pm${model_rot}, T$\pm${model_trans}, Shear$\pm${model_shear}, {model_vers} model\n"
                title_data = f"Performance on $\Theta\pm${data_rot}, T$\pm${data_trans}, Shear$\pm${data_shear} test data"
                fig.suptitle(title_model + title_data, fontsize=16)

                #plt.tight_layout(rect=[0, 0, 1, 0.96])
                save_name = f"dice_plots/data{data_rot}_{data_trans}_{data_shear}_model_baseline_{model_vers}.png"
                #save_name = f"report/images/dice/data{data_rot}_{data_trans}_{data_shear}_model{model_rot}_{model_trans}_{model_shear}_{model_vers}.png"
                plt.savefig (save_name)

                #plt.show()
                print (save_name)
                k += 1
    '''
    for i, (data_rot, data_trans, data_shear) in enumerate (data_combinations):
        if data_shear is None:
            for j, model_vers in enumerate (models, start=8):
                data_folder = f"aug_data/test_norm_rot{data_rot}_trans{data_trans}_shear{data_shear}/"
                #model_folder = f"rot{model_rot}_trans{model_trans}_shear{model_shear}_{model_vers}_model_results/"
                model_folder = f"{model_vers}_results/"
                tag = ((data_rot, data_trans, data_shear), (model_rot, model_trans, model_shear, model_vers))

                dice = np.load (data_folder + model_folder + "segmentations/moved/dice.npy")
                haus = np.load (data_folder + model_folder + "segmentations/moved/hausdorff.npy")
                mean_dice = np.mean (dice, axis=1)
                mean_haus = np.mean (haus, axis=1)
                full_mean_dice = np.mean (mean_dice)
                full_mean_haus = np.mean (mean_haus)
                all_scores[(i,j)] = (mean_dice, mean_haus)
                all_mean_scores[(i,j)] = (full_mean_dice, full_mean_haus)
                #all_mean_scores.append ((i, j, full_mean_dice, full_mean_haus))
                data = [ (label, d, h)
                         for label, d, h
                         in list (zip (segmentation_labels, dice, haus))]
                         #if label != "background"]
                labels, dice, haus = zip (*data)

                #print (dice.shape, mean_dice.shape)
                #print (mean_dice)
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))

                # Violin plot for Dice Coefficient
                dice_violins = axes[0].violinplot(dice, showmeans=True, vert=False)
                axes[0].set_yticks(np.arange(1, len(labels) + 1))
                axes[0].set_yticklabels(labels)
                axes[0].set_xlim(-0.025, 1.025)
                axes[0].set_xlabel("Dice Coefficient")
                axes[0].set_title("Dice Coefficient Distribution")

                # Violin plot for Hausdorff Distance
                haus_violins = axes[1].violinplot(haus, showmeans=True, vert=False)
                axes[1].set_yticks(np.arange(1, len(labels) + 1))
                axes[1].set_yticklabels([])
                axes[1].set_xlabel("Hausdorff Distance")
                axes[1].set_title("Hausdorff Distance Distribution")

                cmap = cm.tab20
                for partname in ("cbars","cmins","cmaxes","cmeans",):
                    violin1 = dice_violins[partname]
                    violin1.set_edgecolor("red")
                    violin1.set_linewidth(1)
                    violin2 = haus_violins[partname]
                    violin2.set_edgecolor("red")
                    violin2.set_linewidth(1)

                #for i, (violin1, violin2) in enumerate(zip(dice_violins["bodies"], haus_violins["bodies"])):
                #    color = cmap(i / len(data))
                #    violin1.set_facecolor(color)
                #    violin1.set_alpha(0.7)
                #    violin2.set_facecolor(color)
                #    violin2.set_alpha(0.7)

                title_model = f"Baseline {model_vers} model\n"
                title_data = f"Performance on $\Theta\pm${data_rot}, T$\pm${data_trans}, Shear$\pm${data_shear} test data"
                fig.suptitle(title_model + title_data, fontsize=16)

                #plt.tight_layout(rect=[0, 0, 1, 0.96])
                #save_name = f"dice_plots/data{data_rot}_{data_trans}_{data_shear}_model_baseline_{model_vers}.png"
                save_name = f"report/images/dice/data{data_rot}_{data_trans}_{data_shear}_model_baseline_{model_vers}.png"
                plt.savefig (save_name)

                #plt.show()
                print (save_name)
                k += 1
    '''
    print (all_mean_scores)
    print ()
    print (combine_by_key (all_mean_scores, 1))
    best_dice = max (combine_by_key (all_mean_scores, 1).items(), key=lambda e: e[1][0])
    white_matter_dice = np.mean (combine_by_key (all_scores, 1)[best_dice[0]][0][white_matter])

    print ("Best model by dice:", best_dice, all_combinations[best_dice[0]])
    #print ("White matter dice:", all_scores[(0,0)][0].shape, all_scores[(0,0)][1].shape)
    print ("Best model White matter dice:", white_matter_dice)
    #print ("Best model by haus:", min (combine_by_key (all_mean_scores, 1).items(), key=lambda e: e[1][1]))
