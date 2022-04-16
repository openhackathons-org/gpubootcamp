from sympy import Symbol, Eq
import numpy as np
import tensorflow as tf

from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, MonitorDomain
from modulus.data import Validation, BC, Monitor
from modulus.sympy_utils.geometry_1d import Point1D
from modulus.controller import ModulusController
from modulus.plot_utils.vtk import var_to_vtk

from spring_mass_ode import SpringMass

# define time variable and range
t_max = 10.0
t_symbol = Symbol('t')
x = Symbol('x')
time_range = {t_symbol: (0, t_max)}

geo = Point1D(0)

deltaT = 0.01
t = np.arange(0, t_max, deltaT)
t = np.expand_dims(t, axis=-1) 

invar_numpy = {'t': t}
outvar_numpy = {'x1': (1/6)*np.cos(t) + (1/2)*np.cos(np.sqrt(3)*t) + (1/3)*np.cos(2*t),
                'x2': (2/6)*np.cos(t) + (0/2)*np.cos(np.sqrt(3)*t) - (1/3)*np.cos(2*t),
                'x3': (1/6)*np.cos(t) - (1/2)*np.cos(np.sqrt(3)*t) + (1/3)*np.cos(2*t)} 
outvar_numpy.update({'ode_x1': np.full_like(invar_numpy['t'], 0)})
outvar_numpy.update({'ode_x2': np.full_like(invar_numpy['t'], 0)})
outvar_numpy.update({'ode_x3': np.full_like(invar_numpy['t'], 0)})

class SpringMassTrain(TrainDomain):
  def __init__(self, **config):
    super(SpringMassTrain, self).__init__()

    # initial conditions
    IC = geo.boundary_bc(outvar_sympy={'x1': 1.,
                                       'x2': 0,
                                       'x3': 0,
                                       'x1__t': 0,
                                       'x2__t': 0,
                                       'x3__t': 0},
                         param_ranges={t_symbol: 0},
                         batch_size_per_area=1)
    self.add(IC, name="IC")

    # data
    data =  BC.from_numpy(invar_numpy, outvar_numpy)

    self.add(data, name="Data")

class SpringMassMonitor(MonitorDomain):
  def __init__(self, **config):
    super(SpringMassMonitor, self).__init__()
    
    global_monitor = Monitor(invar_numpy, {'average_m1': lambda var: tf.reduce_mean(var['m1']),
                                           'average_k4': lambda var: tf.reduce_mean(var['k4'])})
    self.add(global_monitor, 'GlobalMonitor')

class SpringMassSolver(Solver):
  train_domain = SpringMassTrain
  monitor_domain = SpringMassMonitor

  def __init__(self, **config):
    super(SpringMassSolver, self).__init__(**config)

    self.equations = SpringMass(k=(2, 1, 1, 'k4'), m=('m1', 1, 1)).make_node(stop_gradients=['x1', 'x1__t', 'x1__t__t',
                                                                                             'x2', 'x2__t', 'x2__t__t',
                                                                                             'x3', 'x3__t', 'x3__t__t'])

    spring_net = self.arch.make_node(name='spring_net',
                                   inputs=['t'],
                                   outputs=['x1','x2','x3'])
    invert_net = self.arch.make_node(name='invert_net',
                                   inputs=['t'],
                                   outputs=['m1','k4'])

    self.nets = [spring_net, invert_net]

  @classmethod # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_spring_mass_inverse',
        'max_steps': 10000,
        'decay_steps': 100,
        'nr_layers': 6,
        'layer_size': 256,
        'xla': True,
        })


if __name__ == '__main__':
  ctr = ModulusController(SpringMassSolver)
  ctr.run()
