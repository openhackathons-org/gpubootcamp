from sympy import Symbol
import numpy as np
import tensorflow as tf
from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, MonitorDomain
from modulus.data import Validation, Monitor, BC
from modulus.sympy_utils.geometry_2d import Rectangle, Line, Channel2D
from modulus.sympy_utils.functions import parabola
from modulus.csv_utils.csv_rw import csv_to_dict
from modulus.PDES.navier_stokes import IntegralContinuity, NavierStokes
from modulus.controller import ModulusController
from modulus.architecture import FourierNetArch

# TODO: Replace all the placeholders with appropriate values

# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

# OpenFOAM data
mapping = {'Points:0': 'x', 'Points:1': 'y',
           'U:0': 'u', 'U:1': 'v', 'p': 'p'}
openfoam_var = csv_to_dict('openfoam/2D_chip_fluid0.csv', mapping)
openfoam_var['x'] -= 2.5 # normalize pos
openfoam_var['y'] -= 0.5
openfoam_invar_numpy = {key: value for key, value in openfoam_var.items() if key in ['x', 'y']}
openfoam_outvar_numpy = {key: value for key, value in openfoam_var.items() if key in ['u', 'v', 'p']}

#TODO: Add keys and appropriate values for continuity and momentum equations in x and y directions here:
openfoam_outvar_numpy['continuity'] = placeholder
openfoam_outvar_numpy['momentum_x'] = placeholder
openfoam_outvar_numpy['momentum_y'] = placeholder

class Chip2DTrain(TrainDomain):
  def __init__(self, **config):
    super(Chip2DTrain, self).__init__()

    # fill in the appropriate parameters for the from_numpy function
    interior=BC.from_numpy(placeholder, placeholder, batch_size=placeholder)
    self.add(interior, name="Interior")

class Chip2DMonitor(MonitorDomain):
  def __init__(self, **config):
    super(Chip2DMonitor, self).__init__()

    global_monitor = Monitor(openfoam_invar_numpy, {'average_nu': lambda var: tf.reduce_mean(var['nu'])})
    self.add(global_monitor, 'GlobalMonitor')

#TODO: Replace all the placeholders with appropriate values
class ChipSolver(Solver):
  train_domain = placeholder
  monitor_domain = placeholder

  def __init__(self, **config):
    super(ChipSolver, self).__init__(**config)

    self.equations = (NavierStokes(nu=placeholder, rho=1, dim=2, time=False).make_node(stop_gradients=[placeholder]))

    flow_net = self.arch.make_node(name='flow_net',
                                   inputs=['x', 'y'],
                                   outputs=['u', 'v', 'p'])
    invert_net = self.arch.make_node(name='invert_net',
                                     inputs=['x', 'y'],
                                     outputs=['nu'])
    self.nets = [flow_net, invert_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_chip_2d_inverse',
        'rec_results': True,
        'rec_results_freq': 100,
        'start_lr': 3e-4,
        'max_steps': 40000,
        'decay_steps': 100,
        'xla': True
        })
if __name__ == '__main__':
  ctr = ModulusController(ChipSolver)
  ctr.run()
