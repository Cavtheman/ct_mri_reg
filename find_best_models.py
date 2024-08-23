import sys
import numpy as np


if __name__ == "__main__":
    val_paths = sys.argv[1:]

    for val_path in val_paths:
        epoch_vals = np.load (f"{val_path}/val_history.npy")
        min_epoch = np.argmin (epoch_vals)
        print (f"{val_path}/synthmorph_epoch{min_epoch:02d}.h5")
