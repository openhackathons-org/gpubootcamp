from sympy import Symbol, Function, Number
from modulus.eq.pde import PDE


class SpringMass(PDE):
    name = "SpringMass"

    def __init__(self, k=(2, 1, 1, 2), m=(1, 1, 1)):

        self.k = k
        self.m = m

        k1 = k[0]
        k2 = k[1]
        k3 = k[2]
        k4 = k[3]
        m1 = m[0]
        m2 = m[1]
        m3 = m[2]

        t = Symbol("t")
        input_variables = {"t": t}

        x1 = Function("x1")(*input_variables)
        x2 = Function("x2")(*input_variables)
        x3 = Function("x3")(*input_variables)

        if type(k1) is str:
            k1 = Function(k1)(*input_variables)
        elif type(k1) in [float, int]:
            k1 = Number(k1)
        if type(k2) is str:
            k2 = Function(k2)(*input_variables)
        elif type(k2) in [float, int]:
            k2 = Number(k2)
        if type(k3) is str:
            k3 = Function(k3)(*input_variables)
        elif type(k3) in [float, int]:
            k3 = Number(k3)
        if type(k4) is str:
            k4 = Function(k4)(*input_variables)
        elif type(k4) in [float, int]:
            k4 = Number(k4)

        if type(m1) is str:
            m1 = Function(m1)(*input_variables)
        elif type(m1) in [float, int]:
            m1 = Number(m1)
        if type(m2) is str:
            m2 = Function(m2)(*input_variables)
        elif type(m2) in [float, int]:
            m2 = Number(m2)
        if type(m3) is str:
            m3 = Function(m3)(*input_variables)
        elif type(m3) in [float, int]:
            m3 = Number(m3)

        self.equations = {}
        self.equations["ode_x1"] = m1 * (x1.diff(t)).diff(t) + k1 * x1 - k2 * (x2 - x1)
        self.equations["ode_x2"] = (
            m2 * (x2.diff(t)).diff(t) + k2 * (x2 - x1) - k3 * (x3 - x2)
        )
        self.equations["ode_x3"] = m3 * (x3.diff(t)).diff(t) + k3 * (x3 - x2) + k4 * x3
