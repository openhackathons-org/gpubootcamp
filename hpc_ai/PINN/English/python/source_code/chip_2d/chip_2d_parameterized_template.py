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

#TODO: Replace all the placeholders with appropriate values
# you may use the script used in the previous challenge problem as a reference

@modulus.main(config_path="conf", config_name="config_param")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

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
    inlet_vel = 1.5

    # paramteric variables
    chip_height = Symbol("chip_height")  # Not fixed anymore
    chip_width = Symbol("chip_width")    # Not fixed anymore
    chip_height_range = (0.4, 0.8)
    chip_width_range = (0.6, 1.4)
    param_ranges = {chip_height: chip_height_range, chip_width: chip_width_range}

    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")

    # define geometry
    channel = placeholder
    inlet = placeholder
    outlet = placeholder
    rec = placeholder
    flow_rec = placeholder
    geo = channel - rec
    geo_hr = geo & flow_rec
    geo_lr = geo - flow_rec
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
        parameterization=param_ranges,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar=placeholder,
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
        parameterization=placeholder,
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar=placeholder,
        batch_size=cfg.batch_size.no_slip,
        parameterization=placeholder,
    )
    domain.add_constraint(no_slip, "no_slip")
    
    # interior lr
    interior_lr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_lr,
        outvar=placeholder,
        batch_size=cfg.batch_size.interior_lr,
        bounds={x: channel_length, y: channel_width},
        lambda_weighting=placeholder,
        parameterization=placeholder,
    )
    domain.add_constraint(interior_lr, "interior_lr")

    # interior hr
    interior_hr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_hr,
        outvar=placeholder,
        batch_size=cfg.batch_size.interior_hr,
        bounds={x: channel_length, y: channel_width},
        lambda_weighting=placeholder,
        parameterization=placeholder,
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
        outvar=placeholder,
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting=placeholder,
        criteria=integral_criteria,
        parameterization=placeholder,
    )
    domain.add_constraint(integral_continuity, "integral_continuity")
    
  
    # add validation data
    mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    openfoam_var = csv_to_dict(to_absolute_path("openfoam/2D_chip_fluid0.csv"), mapping)
    openfoam_var["x"] -= 2.5  # normalize pos
    openfoam_var["y"] -= 0.5
    
    # TODO: Add the arrays for 'chip_height' and 'chip_width'
    
    openfoam_invar_numpy = {
        key: value
        for key, value in openfoam_var.items()
        if key in ["x", "y", "chip_height", "chip_width"]
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
    
    inference = PointwiseInferencer(
        openfoam_invar_numpy,     # TODO: The invar array can be changed to infer at different geometry combinations
        ["u", "v", "p"],
        nodes,
        batch_size=1024,
    )
    domain.add_inferencer(inference, "inf_data")
    
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
