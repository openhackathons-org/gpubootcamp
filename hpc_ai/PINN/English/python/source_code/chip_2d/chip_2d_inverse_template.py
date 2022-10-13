import torch
import numpy as np
from sympy import Symbol, Eq

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_2d import Rectangle, Line, Channel2D
from modulus.utils.sympy.functions import parabola
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.basic import NormalDotVec
from modulus.domain.constraint import (
    PointwiseConstraint,
)
from modulus.domain.monitor import PointwiseMonitor
from modulus.key import Key
from modulus.node import Node

#TODO: Replace all the placeholders with appropriate values

@modulus.main(config_path="conf", config_name="config_inverse")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu="nu", rho=1.0, dim=2, time=False)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    inverse_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("nu")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        ns.make_nodes(
            detach_names=[placeholder]
        )
        + [flow_net.make_node(name="flow_network", jit=cfg.jit)]
        + [inverse_net.make_node(name="inverse_network", jit=cfg.jit)]
    )

    # add constraints to solver
    # data
    mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    openfoam_var = csv_to_dict(to_absolute_path("openfoam/2D_chip_fluid0.csv"), mapping)
    openfoam_var["x"] -= 2.5  # normalize pos
    openfoam_var["y"] -= 0.5
    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    }
    openfoam_outvar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["u", "v", "p"]
    }
    openfoam_outvar_numpy["continuity"] = placeholder
    openfoam_outvar_numpy["momentum_x"] = placeholder
    openfoam_outvar_numpy["momentum_y"] = placeholder

    # make domain
    domain = Domain()

    # data and pdes
    data = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=openfoam_invar_numpy,
        outvar=openfoam_outvar_numpy,
        batch_size=cfg.batch_size.data,
    )
    domain.add_constraint(data, name="Data")
    
    # add monitors
    monitor = PointwiseMonitor(
        openfoam_invar_numpy,
        output_names=["nu"],
        metrics={"mean_nu": lambda var: torch.mean(var["nu"])},
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