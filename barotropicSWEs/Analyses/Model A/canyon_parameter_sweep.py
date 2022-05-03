#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Wed Mar 16 12:55:08 2022

"""
import numpy as np


def vary_parameter(
    param,
    bbox,
    case,
    parameter_range=(2, 20),  # Dimensional parameter range in either km or m
    number_of_runs=6,
    order=3,
    numerical_flux="Central",
    slope="cos_squared",
):

    assert case.upper() in [
        "WIDTH",
        "LENGTH",
        "DEPTH",
    ], "Invalid choice of \
parameter on which to conduct parameter sweep. Must be either 'width', 'depth'\
 or 'length'."

    assert numerical_flux.upper() in [
        "CENTRAL",
        "PENALTY",
        "UPWIND",
        "LAX-FRIEDRICHS" "ALTERNATING",
    ], "Invalid choice of numerical flux for DG-FEM method. Must be either \
'Central', 'Penalty', 'Upwind', 'Lax-Friedrichs' or 'Altnerating'."

    parameter_values = np.linspace(
        parameter_range[0], parameter_range[1], number_of_runs
    )

    if case.upper() == "WIDTH":
        from parameter_runs import width_run

        print("ere")
        width_run(param, bbox, parameter_values, order, numerical_flux, slope)

    elif case.upper() == "LENGTH":
        pass

    else:  # Must be "Depth"
        pass


if __name__ == "__main__":
    from barotropicSWEs.Configuration import configure

    param = configure.main()
    bbox = (-50, 50, 0, 100)
    vary_parameter(
        param,  # Default parameter values stored here
        bbox,  # Domain boundary box given in km
        case="Width",  # Test cases ['width', 'depth', 'length'],
        parameter_range=(0, 20),  # Dimensional parameter range in either km or m
        number_of_runs=21,
        order=5,  # Integer value > 0 used in h-refinement,
        numerical_flux="Central",
        slope="cos_squared",  # Slope in ['cos_squared', 'gg07']
    )
