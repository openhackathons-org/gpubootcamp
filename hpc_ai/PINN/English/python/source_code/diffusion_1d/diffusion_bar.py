# import SimNet library
from sympy import Symbol, sin, Eq, Abs, exp
import numpy as np
import sys
sys.path.append('../../')
from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, MonitorDomain
from simnet.data import Validation, BC, Monitor
from simnet.sympy_utils.geometry_1d import Line1D
#from simnet.PDES.diffusion import Diffusion
from simnet.controller import SimNetController
from simnet.node import Node
from simnet.pdes import PDES
import tensorflow as tf
from sympy import Symbol, Function, Number

from simnet.pdes import PDES
from simnet.node import Node
from simnet.variables import Variables


# params for domain
L1 = Line1D(0,1)
L2 = Line1D(1,2)

D1 = 1e1
D2 = 1e-1

Tc = 100
Ta = 0
Tb = (Tc + (D1/D2)*Ta)/(1 + (D1/D2))

#Tb = Tc - u_1__x*D1/D2 
#Ta = Tb - u_1__x
print(Ta)
print(Tb)
print(Tc)
#exit()

class Diffusion(PDES):
  name = 'Diffusion'
 
  def __init__(self, T='T', D='D', Q=0, dim=3, time=True):
    # set params
    self.T = T
    self.dim = dim
    self.time = time

    # coordinates
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

    # time
    t = Symbol('t')

    # make input variables 
    input_variables = {'x':x,'y':y,'z':z,'t':t}
    if self.dim == 1:
      input_variables.pop('y')
      input_variables.pop('z')
    elif self.dim == 2:
      input_variables.pop('z')
    if not self.time: 
      input_variables.pop('t')

    # Temperature
    assert type(T) == str, "T needs to be string"
    T = Function(T)(*input_variables)

    # Diffusivity
    if type(D) is str:
      D = Function(D)(*input_variables)
    elif type(D) in [float, int]:
      D = Number(D)

    # Source
    if type(Q) is str:
      Q = Function(Q)(*input_variables)
    elif type(Q) in [float, int]:
      Q = Number(Q)

    # set equations
    self.equations = Variables()
    self.equations['diffusion_'+self.T] = (T.diff(t)
                                            - (D*T.diff(x)).diff(x)
                                            - (D*T.diff(y)).diff(y)
                                            - (D*T.diff(z)).diff(z)
                                            - Q)

class DiffusionInterface(PDES):
  name = 'DiffusionInterface'

  def __init__(self, T_1, T_2, D_1, D_2, dim=3, time=True):
    # set params
    self.T_1 = T_1
    self.T_2 = T_2
    self.D_1 = D_1
    self.D_2 = D_2
    self.dim = dim
    self.time = time
 
    # coordinates
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    normal_x, normal_y, normal_z = Symbol('normal_x'), Symbol('normal_y'), Symbol('normal_z')

    # time
    t = Symbol('t')

    # make input variables 
    input_variables = {'x':x,'y':y,'z':z,'t':t}
    if self.dim == 1:
      input_variables.pop('y')
      input_variables.pop('z')
    elif self.dim == 2:
      input_variables.pop('z')
    if not self.time: 
      input_variables.pop('t')

    # variables to match the boundary conditions (example Temperature)
    T_1 = Function(T_1)(*input_variables)
    T_2 = Function(T_2)(*input_variables)

    # set equations
    self.equations = Variables()
    self.equations['diffusion_interface_dirichlet_'+self.T_1+'_'+self.T_2] = T_1 - T_2
    flux_1 = self.D_1 * (normal_x * T_1.diff(x) + normal_y * T_1.diff(y) + normal_z * T_1.diff(z))
    flux_2 = self.D_2 * (normal_x * T_2.diff(x) + normal_y * T_2.diff(y) + normal_z * T_2.diff(z))
    self.equations['diffusion_interface_neumann_'+self.T_1+'_'+self.T_2] = flux_1 - flux_2

class DiffusionTrain(TrainDomain):
  def __init__(self, **config):
    super(DiffusionTrain, self).__init__()
    # sympy variables
    x = Symbol('x')
    c = Symbol('c')
    
    # right hand side (x = 2) Pt c
    IC = L2.boundary_bc(outvar_sympy={'u_2': Tc},
                        batch_size_per_area=1,
                        criteria=Eq(x, 2))
    self.add(IC, name="RightHandSide")
    
    # left hand side (x = 0) Pt a
    IC = L1.boundary_bc(outvar_sympy={'u_1': Ta},
                        batch_size_per_area=1,
                        criteria=Eq(x, 0))
    self.add(IC, name="LeftHandSide")
    
    # interface 1-2
    IC = L1.boundary_bc(outvar_sympy={'diffusion_interface_dirichlet_u_1_u_2': 0,
                                      'diffusion_interface_neumann_u_1_u_2': 0},
                        lambda_sympy={'lambda_diffusion_interface_dirichlet_u_1_u_2': 1,
                                      'lambda_diffusion_interface_neumann_u_1_u_2': 1},
                        batch_size_per_area=1,
                        criteria=Eq(x, 1))
    self.add(IC, name="Interface1n2")
    
    # interior 1
    interior = L1.interior_bc(outvar_sympy={'diffusion_u_1': 0},
                              lambda_sympy={'lambda_diffusion_u_1': 1},
                              bounds={x: (0, 1)},
                              batch_size_per_area=200)
    self.add(interior, name="Interior1")
    
    # interior 2
    interior = L2.interior_bc(outvar_sympy={'diffusion_u_2': 0},
                              lambda_sympy={'lambda_diffusion_u_2': 1},
                              bounds={x: (1, 2)},
                              batch_size_per_area=200)
    self.add(interior, name="Interior2")

class DiffusionVal(ValidationDomain):
  def __init__(self, **config):
    super(DiffusionVal, self).__init__()
    # make validation data line 1
    #Tc = u_2__2
    #Tb = Tc - u_1__x*D1/D2 
    #Ta = Tb - u_1__x

    x = np.expand_dims(np.linspace(0, 1, 100), axis=-1)
    u_1 = x*Tb + (1-x)*Ta
    invar_numpy = {'x': x}
    outvar_numpy = {'u_1': u_1}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val1')
    
    # make validation data line 2
    x = np.expand_dims(np.linspace(1, 2, 100), axis=-1)
    u_2 = (x-1)*Tc + (2-x)*Tb
    invar_numpy = {'x': x}
    outvar_numpy = {'u_2': u_2}
    val = Validation.from_numpy(invar_numpy, outvar_numpy)
    self.add(val, name='Val2')

class DiffusionMonitor(MonitorDomain):
  def __init__(self, **config):
    super(DiffusionMonitor, self).__init__()
    x = Symbol('x')

    # flux in U1 at x = 1 
    fluxU1 = Monitor(L1.sample_boundary(10, criteria=Eq(x, 1)),
                    {'flux_U1': lambda var: tf.reduce_mean(D1*var['u_1__x'])})
    self.add(fluxU1, 'FluxU1')

    # flux in U2 at x = 1 
    fluxU2 = Monitor(L2.sample_boundary(10, criteria=Eq(x, 1)),
                    {'flux_U2': lambda var: tf.reduce_mean(D2*var['u_2__x'])})
    self.add(fluxU2, 'FluxU2')

# Define neural network
class DiffusionSolver(Solver):
  train_domain = DiffusionTrain
  val_domain = DiffusionVal
  monitor_domain = DiffusionMonitor

  def __init__(self, **config):
    super(DiffusionSolver, self).__init__(**config)

    self.equations = (Diffusion(T='u_1', D=D1, dim=1, time=False).make_node()
                      + Diffusion(T='u_2', D=D2, dim=1, time=False).make_node()
                      + DiffusionInterface('u_1', 'u_2', D1, D2, dim=1, time=False).make_node())
    diff_net_u_1 = self.arch.make_node(name='diff_net_u_1',
                                   inputs=['x'],
                                   outputs=['u_1'])
    diff_net_u_2 = self.arch.make_node(name='diff_net_u_2',
                                   inputs=['x'],
                                   outputs=['u_2'])
    self.nets = [diff_net_u_1, diff_net_u_2]

  @classmethod # Explain This
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_diff',
        'max_steps': 5000,
        'decay_steps': 100,
        'start_lr': 1e-4,
        'xla': True,
        #'end_lr': 1e-6,
        })

if __name__ == '__main__':
  ctr = SimNetController(DiffusionSolver)
  ctr.run()
