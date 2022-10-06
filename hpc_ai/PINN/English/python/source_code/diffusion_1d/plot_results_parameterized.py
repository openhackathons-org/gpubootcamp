import numpy as np
import matplotlib.pyplot as plt

network_dir = "./outputs/diffusion_bar_parameterized/validators/"
data_1 = np.load(network_dir + "Val1.npz", allow_pickle=True)
data_2 = np.load(network_dir + "Val2.npz", allow_pickle=True)
data_1 = np.atleast_1d(data_1.f.arr_0)[0]
data_2 = np.atleast_1d(data_2.f.arr_0)[0]

plt.plot(data_1["x"][:, 0], data_1["pred_u_1"][:, 0], "--", label="u_1_pred")
plt.plot(data_2["x"][:, 0], data_2["pred_u_2"][:, 0], "--", label="u_2_pred")
plt.plot(data_1["x"][:, 0], data_1["true_u_1"][:, 0], label="u_1_true")
plt.plot(data_2["x"][:, 0], data_2["true_u_2"][:, 0], label="u_2_true")

plt.legend()
plt.savefig("image_diffusion_problem_bootcamp_parameterized")
