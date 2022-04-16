from sympy import Symbol, Eq
import numpy as np

from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain
from modulus.data import Validation
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

    # solve over given time period
    interior = geo.boundary_bc(outvar_sympy={'ode_x1': 0.0,
                                             'ode_x2': 0.0,
                                             'ode_x3': 0.0}, 
                               param_ranges=time_range,
                               batch_size_per_area=500)
    self.add(interior, name="Interior")

class SpringMassVal(ValidationDomain):
   def __init__(self, **config):
     super(SpringMassVal, self).__init__()
     deltaT = 0.001
     t = np.arange(0, t_max, deltaT)
     t = np.expand_dims(t, axis=-1) 
     invar_numpy = {'t': t}
     outvar_numpy = {'x1': (1/6)*np.cos(t) + (1/2)*np.cos(np.sqrt(3)*t) + (1/3)*np.cos(2*t),
                     'x2': (2/6)*np.cos(t) + (0/2)*np.cos(np.sqrt(3)*t) - (1/3)*np.cos(2*t),
                     'x3': (1/6)*np.cos(t) - (1/2)*np.cos(np.sqrt(3)*t) + (1/3)*np.cos(2*t)} 
     val = Validation.from_numpy(invar_numpy, outvar_numpy)
     self.add(val, name="Val")

class SpringMassSolver(Solver):
  train_domain = SpringMassTrain
  val_domain = SpringMassVal

  def __init__(self, **config):
    super(SpringMassSolver, self).__init__(**config)

    self.equations = SpringMass(k=(2, 1, 1, 2), m=(1, 1, 1)).make_node()

    spring_net = self.arch.make_node(name='spring_net',
                                   inputs=['t'],
                                   outputs=['x1','x2','x3'])
    self.nets = [spring_net]

  @classmethod # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_spring_mass',
        'max_steps': 10000,
        'decay_steps': 100,
        'nr_layers': 6,
        'layer_size': 256,
        'xla': True,
        })


if __name__ == '__main__':
  ctr = ModulusController(SpringMassSolver)
  ctr.run()
