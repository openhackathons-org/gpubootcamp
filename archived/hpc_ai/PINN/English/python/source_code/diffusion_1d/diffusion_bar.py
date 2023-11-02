import torch
import numpy as np
from sympy import Symbol, Eq, Function, Number

import modulus
from modulus.hydra import instantiate_arch , ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_1d import Line1D
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.domain.validator import PointwiseValidator
from modulus.domain.monitor import PointwiseMonitor
from modulus.key import Key
from modulus.node import Node

from modulus.eq.pde import PDE


# params for domain
L1 = Line1D(0, 1)
L2 = Line1D(1, 2)

D1 = 1e1
D2 = 1e-1

Tc = 100
Ta = 0
Tb = (Tc + (D1 / D2) * Ta) / (1 + (D1 / D2))

print(Ta)
print(Tb)
print(Tc)


class Diffusion(PDE):
    name = "Diffusion"

    def __init__(self, T="T", D="D", Q=0, dim=3, time=True):
        # set params
        self.T = T
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

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
        self.equations = {}
        self.equations["diffusion_" + self.T] = (
            T.diff(t)
            - (D * T.diff(x)).diff(x)
            - (D * T.diff(y)).diff(y)
            - (D * T.diff(z)).diff(z)
            - Q
        )


class DiffusionInterface(PDE):
    name = "DiffusionInterface"

    def __init__(self, T_1, T_2, D_1, D_2, dim=3, time=True):
        # set params
        self.T_1 = T_1
        self.T_2 = T_2
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x, normal_y, normal_z = (
            Symbol("normal_x"),
            Symbol("normal_y"),
            Symbol("normal_z"),
        )

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Diffusivity
        if type(D_1) is str:
            D_1 = Function(D_1)(*input_variables)
        elif type(D_1) in [float, int]:
            D_1 = Number(D_1)
        if type(D_2) is str:
            D_2 = Function(D_2)(*input_variables)
        elif type(D_2) in [float, int]:
            D_2 = Number(D_2)

        # variables to match the boundary conditions (example Temperature)
        T_1 = Function(T_1)(*input_variables)
        T_2 = Function(T_2)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["diffusion_interface_dirichlet_" + self.T_1 + "_" + self.T_2] = (
            T_1 - T_2
        )
        flux_1 = D_1 * (
            normal_x * T_1.diff(x) + normal_y * T_1.diff(y) + normal_z * T_1.diff(z)
        )
        flux_2 = D_2 * (
            normal_x * T_2.diff(x) + normal_y * T_2.diff(y) + normal_z * T_2.diff(z)
        )
        self.equations["diffusion_interface_neumann_" + self.T_1 + "_" + self.T_2] = (
            flux_1 - flux_2
        )


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    diff_u1 = Diffusion(T="u_1", D=D1, dim=1, time=False)
    diff_u2 = Diffusion(T="u_2", D=D2, dim=1, time=False)
    diff_in = DiffusionInterface("u_1", "u_2", D1, D2, dim=1, time=False)

    diff_net_u_1 = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u_1")],
        cfg=cfg.arch.fully_connected,
    )
    diff_net_u_2 = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u_2")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        diff_u1.make_nodes()
        + diff_u2.make_nodes()
        + diff_in.make_nodes()
        + [diff_net_u_1.make_node(name="u1_network", jit=cfg.jit)]
        + [diff_net_u_2.make_node(name="u2_network", jit=cfg.jit)]
    )

    # make domain add constraints to the solver
    domain = Domain()

    # sympy variables
    x = Symbol("x")

    # right hand side (x = 2) Pt c
    rhs = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=L2,
        outvar={"u_2": Tc},
        batch_size=cfg.batch_size.rhs,
        criteria=Eq(x, 2),
    )
    domain.add_constraint(rhs, "right_hand_side")

    # left hand side (x = 0) Pt a
    lhs = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=L1,
        outvar={"u_1": Ta},
        batch_size=cfg.batch_size.lhs,
        criteria=Eq(x, 0),
    )
    domain.add_constraint(lhs, "left_hand_side")

    # interface 1-2
    interface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=L1,
        outvar={
            "diffusion_interface_dirichlet_u_1_u_2": 0,
            "diffusion_interface_neumann_u_1_u_2": 0,
        },
        batch_size=cfg.batch_size.interface,
        criteria=Eq(x, 1),
    )
    domain.add_constraint(interface, "interface")

    # interior 1
    interior_u1 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=L1,
        outvar={"diffusion_u_1": 0},
        bounds={x: (0, 1)},
        batch_size=cfg.batch_size.interior_u1,
    )
    domain.add_constraint(interior_u1, "interior_u1")

    # interior 2
    interior_u2 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=L2,
        outvar={"diffusion_u_2": 0},
        bounds={x: (1, 2)},
        batch_size=cfg.batch_size.interior_u2,
    )
    domain.add_constraint(interior_u2, "interior_u2")

    # validation data
    x = np.expand_dims(np.linspace(0, 1, 100), axis=-1)
    u_1 = x * Tb + (1 - x) * Ta
    invar_numpy = {"x": x}
    outvar_numpy = {"u_1": u_1}
    val = PointwiseValidator(nodes=nodes,invar=invar_numpy, true_outvar=outvar_numpy)
    domain.add_validator(val, name="Val1")

    # make validation data line 2
    x = np.expand_dims(np.linspace(1, 2, 100), axis=-1)
    u_2 = (x - 1) * Tc + (2 - x) * Tb
    invar_numpy = {"x": x}
    outvar_numpy = {"u_2": u_2}
    val = PointwiseValidator(nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy)
    domain.add_validator(val, name="Val2")

    # make monitors
    invar_numpy = {"x": [[1.0]]}
    monitor = PointwiseMonitor(
        invar_numpy,
        output_names=["u_1__x"],
        metrics={"flux_u1": lambda var: torch.mean(var["u_1__x"])},
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(monitor)

    monitor = PointwiseMonitor(
        invar_numpy,
        output_names=["u_2__x"],
        metrics={"flux_u2": lambda var: torch.mean(var["u_2__x"])},
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(monitor)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
