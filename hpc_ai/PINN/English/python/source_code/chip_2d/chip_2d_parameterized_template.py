from sympy import Symbol
import numpy as np
import tensorflow as tf
from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, InferenceDomain
from modulus.data import Validation, Inference
from modulus.sympy_utils.geometry_2d import Rectangle, Line, Channel2D
from modulus.sympy_utils.functions import parabola
from modulus.csv_utils.csv_rw import csv_to_dict
from modulus.PDES.navier_stokes import IntegralContinuity, NavierStokes
from modulus.controller import ModulusController
from modulus.architecture import FourierNetArch
from modulus.learning_rate import ExponentialDecayLRWithWarmup

# simulation params
channel_length = (-2.5, 2.5)
channel_width = (-0.5, 0.5)
chip_pos = -1.0
#chip_height = 0.6         # Not fixed anymore
#chip_width = 1.0          # Not fixed anymore
inlet_vel = 1.5

# paramteric variables
chip_height = Symbol('chip_height')
chip_width = Symbol('chip_width')

chip_height_range = (0.4, 0.8)
chip_width_range  = (0.6, 1.4)

param_ranges = {chip_height: chip_height_range,
                chip_width: chip_width_range}

# TODO: Replace all the placeholders with appropriate geometry constructions
# define geometry here
# you may use the geometry generated in the previous challenge problem as a reference

channel = placeholder
# define inlet and outlet
inlet = placeholder
outlet = placeholder
# define the chip
rec = placeholder
# create a geometry for higher sampling of point cloud near the fin
flow_rec = placeholder

# fluid area
geo = placeholder
geo_hr = placeholder
geo_lr = placeholder

x_pos = Symbol('x_pos')
integral_line = placeholder
x_pos_range = {x_pos: lambda batch_size: np.full((batch_size, 1), np.random.uniform(channel_length[0], channel_length[1]))}

# TODO: Replace all the placeholders with appropriate values

# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

class Chip2DTrain(TrainDomain):
  def __init__(self, **config):
    super(Chip2DTrain, self).__init__()

    # inlet
    inlet_parabola = parabola(y, channel_width[0], channel_width[1], inlet_vel)
    inlet_bc = inlet.boundary_bc(outvar_sympy={'u': inlet_parabola, 'v': 0},
                                 batch_size_per_area=64,
                                 param_ranges=param_ranges)
    self.add(inlet_bc, name="Inlet")

    # outlet
    outlet_bc = outlet.boundary_bc(outvar_sympy={placeholder},
                                   batch_size_per_area=placeholder,
                                   param_ranges=placeholder)
    self.add(outlet_bc, name="Outlet")

    # noslip
    noslip = geo.boundary_bc(outvar_sympy={placeholder},
                             batch_size_per_area=placeholder,
                             param_ranges=placeholder)
    self.add(noslip, name="ChipNS")

    # interior lr
    interior_lr = geo_lr.interior_bc(outvar_sympy={placeholder},
                                     bounds={placeholder},
                                     lambda_sympy={placeholder},
                                     batch_size_per_area=placeholder,
                                     param_ranges=placeholder)
    self.add(interior_lr, name="InteriorLR")

    # interior hr
    interior_hr = geo_hr.interior_bc(outvar_sympy={placeholder},
                                     bounds={placeholder},
                                     lambda_sympy={placeholder},
                                     batch_size_per_area=placeholder,
                                     param_ranges=placeholder)
    self.add(interior_hr, name="InteriorHR")


    # integral continuity
    for i in range(4):
      IC = integral_line.boundary_bc(outvar_sympy={placeholder},
                                     batch_size_per_area=placeholder,
                                     lambda_sympy={placeholder},
                                     criteria=placeholder,
                                     param_ranges={placeholder},
                                     fixed_var=placeholder)
      self.add(IC, name="IntegralContinuity_"+str(i))

# validation data
mapping = {'Points:0': 'x', 'Points:1': 'y',
           'U:0': 'u', 'U:1': 'v', 'p': 'p'}
openfoam_var = csv_to_dict('openfoam/2D_chip_fluid0.csv', mapping)
openfoam_var['x'] -= 2.5 # normalize pos
openfoam_var['y'] -= 0.5

# TODO: Add the arrays for 'chip_height' and 'chip_width'

openfoam_invar_numpy = {key: value for key, value in openfoam_var.items() if key in ['x', 'y', 'chip_height', 'chip_width']}
openfoam_outvar_numpy = {key: value for key, value in openfoam_var.items() if key in ['u', 'v', 'p']}

class Chip2DVal(ValidationDomain):
  def __init__(self, **config):
    super(Chip2DVal, self).__init__()
    val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
    self.add(val, name='Val')

class Chip2DInf(InferenceDomain):
  def __init__(self, **config):
    super(Chip2DInf, self).__init__()
    inf = Inference(geo.sample_interior(2048, bounds={x: channel_length, y: channel_width}, 
                                        param_ranges={chip_height: 0.4, chip_width: 1.4}),
                    ['u', 'v', 'p'])
    self.add(inf, name='Inference')

#TODO: Replace all the placeholders with appropriate values
class ChipSolver(Solver):
  train_domain = placeholder
  val_domain = placeholder
  arch = FourierNetArch
  lr = ExponentialDecayLRWithWarmup
  inference_domain = placeholder

  def __init__(self, **config):
    super(ChipSolver, self).__init__(**config)

    self.frequencies = ('axis,diagonal', [i/5. for i in range(25)]) 

    self.equations = (placeholder)
    flow_net = self.arch.make_node(name='flow_net',
                                   inputs=[placeholder],
                                   outputs=[placeholder])
    self.nets = [flow_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_chip_2d_parameterized',
        'rec_results': True,
        'rec_results_freq': 5000,
        'max_steps': 20000,
        'decay_steps': 400,
        'warmup_type': 'gradual',
        'warmup_steps': 2000,
        'xla': True
        })
if __name__ == '__main__':
  ctr = ModulusController(ChipSolver)
  ctr.run()

