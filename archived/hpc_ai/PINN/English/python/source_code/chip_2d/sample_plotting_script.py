# template script to create some easy plots for the chip problem
import numpy as np
import matplotlib.pyplot as plt

import simnet as sn

# set the path for the .npz files
base_dir = 'network_checkpoint_chip_2d/val_domain/results/'

# load the .npz files
pred_data = np.load(base_dir + 'Val_pred.npz', allow_pickle=True)
true_data = np.load(base_dir + 'Val_true.npz', allow_pickle=True)

pred_data = np.atleast_1d(pred_data.f.arr_0)[0]
true_data = np.atleast_1d(true_data.f.arr_0)[0]

# remove the variables created for parameterization (uncomment when visualizing parameteric results)
#pred_data.pop('chip_width')
#pred_data.pop('chip_height')
#true_data.pop('chip_width')
#true_data.pop('chip_height')

# plot only one set of variables
sn.plot_utils.field.plot_field(pred_data, 'chip_predicted', coordinates=['x', 'y'], resolution=256)

# plot the comparison between a set of variables
sn.plot_utils.field.plot_field_compare(true_data, pred_data, 'chip_comparison', coordinates=['x', 'y'], resolution=256)
