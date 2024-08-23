import re
import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_paths = sys.argv[1:]
    save_folder = "report/images/"
    print (data_paths)
    for folder in data_paths:
        print ("Looking at:", folder)
        pattern = r"rot(?P<rot>[0-9.]+)_trans(?P<trans>[0-9.]+)_shear(?P<shear>[0-9.]+|None)/finetune_(?P<model>rigid|affine)"
        match = re.search(pattern, folder)
        if match:
            rot = match.group("rot")
            trans = match.group("trans")
            shear = match.group("shear")
            model = match.group("model")
            print (rot, trans, shear, model)

            loss = np.load (f"{folder}/loss_history.npy")
            val_loss = np.load (f"{folder}/val_history.npy")

            print (np.mean (loss), np.mean (val_loss))
            plt.figure()
            plt.xticks(np.arange(1, len(loss)+1, 1))
            plt.plot(range(1, len(loss)+1), loss, linestyle="-", label="Training Loss")
            plt.plot(range(1, len(val_loss)+1), val_loss, linestyle="-", label="Validation Loss")
            plt.title(f"Training loss of {model} model with parameters:\n$\Theta$=$\pm${rot}, t=$\pm${trans}, shear=$\pm${shear}")
            plt.xlabel("Epoch")
            plt.ylabel("$\|A\hat{B} - I\|$")
            plt.legend()
            #plt.show()
            plt.savefig(save_folder + f"loss_{rot}_{trans}_{shear}_{model}.png")
