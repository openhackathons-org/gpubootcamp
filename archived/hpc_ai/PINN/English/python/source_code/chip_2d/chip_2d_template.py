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
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    #TODO: Replace all the placeholders with appropriate values
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.02, rho=1.0, dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[placeholder],
        output_keys=[placeholder],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network", jit=cfg.jit)]
    )


    # add constraints to solver
    # simulation params
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    chip_pos = -1.0
    chip_height = 0.6
    chip_width = 1.0
    inlet_vel = 1.5

    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")

    #TODO: Replace x1, y1, x2, y2, and X's with appropriate values

    # define geometry
    channel = Channel2D(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )
    inlet = Line(
        (x1, y1),
        (x1, y2),
        normal=1,
    )
    outlet = Line(
        (x2, y1),
        (x2, y2),
        normal=1,
    )
    rec = Rectangle(
        (x1, y1),
        (x2, y2),
    )
    flow_rec = Rectangle(
        (chip_pos - 0.25, channel_width[0]),
        (chip_pos + chip_width + 0.25, channel_width[1]),
    )
    geo = channel - rec
    geo_hr = geo & flow_rec
    geo_lr = geo - flow_rec

    # Optional integral continuity planes to speed up convergence
    x_pos = Symbol("x_pos")
    integral_line = Line((x_pos, channel_width[0]), (x_pos, channel_width[1]), 1)
    x_pos_range = {
        x_pos: lambda batch_size: np.full(
            (batch_size, 1), np.random.uniform(channel_length[0], channel_length[1])
        )
    }

    #TODO: Replace all the placeholders with appropriate values

    # make domain
    domain = Domain()

    # inlet
    inlet_parabola = parabola(y, channel_width[0], channel_width[1], inlet_vel)
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": inlet_parabola, "v": 0},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={placeholder},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={placeholder},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")


    # interior lr
    interior_lr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_lr,
        outvar={placeholder},
        batch_size=cfg.batch_size.interior_lr,
        bounds={x: channel_length, y: channel_width},
        lambda_weighting={placeholder},
    )
    domain.add_constraint(interior_lr, "interior_lr")

    # interior hr
    interior_hr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_hr,
        outvar={placeholder},
        batch_size=cfg.batch_size.interior_hr,
        bounds={x: channel_length, y: channel_width},
        lambda_weighting={placeholder},
    )
    domain.add_constraint(interior_hr, "interior_hr")

    # integral continuity
    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    # integral continuity
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel": 1},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 1},
        criteria=integral_criteria,
        parameterization=x_pos_range,
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    #TODO: Set the appropriate normalization for the validation data
    # The validation data has domain extents of (0,0) to (5,1). Normalize this based on your definition of the domain

    # add validation data
    mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    openfoam_var = csv_to_dict(to_absolute_path("openfoam/2D_chip_fluid0.csv"), mapping)
    openfoam_var["x"] -= placeholder  # TODO normalize pos
    openfoam_var["y"] -= placeholder  # TODO normalize pos
    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    }
    openfoam_outvar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["u", "v", "p"]
    }
    openfoam_validator = PointwiseValidator(
        nodes=nodes,
        invar=openfoam_invar_numpy,
        true_outvar=openfoam_outvar_numpy,
    )
    domain.add_validator(openfoam_validator)


    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
