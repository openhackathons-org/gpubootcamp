
import numpy as np
import matplotlib.pyplot as plt

import simnet as sn

base_dir = './network_checkpoint_spring_mass_pointMass_2/val_domain/results/'

# plot in 1d
predicted_data = np.load(base_dir + 'Val_pred.npz', allow_pickle=True)
true_data = np.load(base_dir + 'Val_true.npz', allow_pickle=True)
true_data = np.atleast_1d(true_data.f.arr_0)[0]
predicted_data = np.atleast_1d(predicted_data.f.arr_0)[0]

print(predicted_data)
print(true_data)

plt.plot(true_data['t'], true_data['x1'], label='True x1')
plt.plot(true_data['t'], true_data['x2'], label='True x2')
plt.plot(true_data['t'], true_data['x3'], label='True x3')
plt.plot(predicted_data['t'], predicted_data['x1'], label='Pred x1')
plt.plot(predicted_data['t'], predicted_data['x2'], label='Pred x2')
plt.plot(predicted_data['t'], predicted_data['x3'], label='Pred x3')
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.legend()
plt.savefig("comparison_new.png")

