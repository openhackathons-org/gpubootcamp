
import numpy as np
import matplotlib.pyplot as plt

network_dir = './network_checkpoint_diff/val_domain/results/'

u_1_pred = np.load(network_dir + 'Val1_pred.npz', allow_pickle=True)
u_2_pred = np.load(network_dir + 'Val2_pred.npz', allow_pickle=True)
u_1_pred = np.atleast_1d(u_1_pred.f.arr_0)[0]
u_2_pred = np.atleast_1d(u_2_pred.f.arr_0)[0]

plt.plot(u_1_pred['x'][:,0], u_1_pred['u_1'][:,0], '--', label='u_1_pred')
plt.plot(u_2_pred['x'][:,0], u_2_pred['u_2'][:,0], '--', label='u_2_pred')

u_1_true = np.load(network_dir + 'Val1_true.npz', allow_pickle=True)
u_2_true = np.load(network_dir + 'Val2_true.npz', allow_pickle=True)
u_1_true = np.atleast_1d(u_1_true.f.arr_0)[0]
u_2_true = np.atleast_1d(u_2_true.f.arr_0)[0]

plt.plot(u_1_true['x'][:,0], u_1_true['u_1'][:,0], label='u_1_true')
plt.plot(u_2_true['x'][:,0], u_2_true['u_2'][:,0], label='u_2_true')

plt.legend()
plt.savefig('image_diffusion_problem_bootcamp')

