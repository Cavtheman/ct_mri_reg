import os
import sys
import torch
import numpy as np

from scipy.linalg import norm

'''
find . -type d -path "./aug_data/*/transforms" -exec python metric_test.py {} \;

./aug_data/rot0.2_trans20_shearNone/transforms
37.83109935877306

./aug_data/rotNone_transNone_shearNone/transforms
0.0

./aug_data/norm_rot0.2_trans20_shear0.1/transforms
38.91170695354462

./aug_data/rot0.4_trans40_shearNone/transforms
78.1706141729249

./aug_data/norm_rot0.2_trans20_shearNone/transforms
38.97704529596379

./aug_data/rot0.2_trans20_shear0.1/transforms
41.275138115485504
'''
if __name__ == "__main__":
    transform_path = sys.argv[1]
    print (transform_path)
    all_transform_norms = [ norm (torch.load (transform_path + f"/{img}").numpy() - np.eye(4,4))
                            for img in os.listdir (transform_path) ]
    #print (all_transform_norms)
    print (np.mean (all_transform_norms))
