import numpy as np
import matplotlib.pyplot as plt

base_dir = "outputs/spring_mass_solver/validators/"

# plot in 1d
data = np.load(base_dir + "validator.npz", allow_pickle=True)
data = np.atleast_1d(data.f.arr_0)[0]

plt.plot(data["t"], data["true_x1"], label="True x1")
plt.plot(data["t"], data["true_x2"], label="True x2")
plt.plot(data["t"], data["true_x3"], label="True x3")
plt.plot(data["t"], data["pred_x1"], label="Pred x1")
plt.plot(data["t"], data["pred_x2"], label="Pred x2")
plt.plot(data["t"], data["pred_x3"], label="Pred x3")
plt.legend()
plt.savefig("comparison.png")
