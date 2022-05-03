#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Fri Feb 25 11:54:03 2022

"""
import numpy as np

def square_eigenvalue_problem(param,
                              order=1,
                              length=1,
                              h_target=.25,
                              tolerance=.1,
                              numerical_fluxes='Central',
                              theta_value=.5,
                              error_base=10,
                              verbose=False
                              ):
    from ppp.File_Management import dir_assurer, file_exist
    import pickle
    from DGFEM.dgfem import FEM
    
    if type(numerical_fluxes)==str:
        numerical_fluxes = [numerical_fluxes]

    bbox = (0, length, 0, length)
    x0, xN, y0, yN = bbox
    folder_name_ = f'Eigenvalue_Problem/NonRotating/h={h_target:.2f}'
    dir_assurer(folder_name_)
    h_func = np.vectorize(lambda X, Y : param.H_D)

    from barotropicSWEs.MeshGeneration import uniform_box_mesh
    P, T = uniform_box_mesh.main(bbox,
                                 (1-tolerance) * h_target,
                                 (1+tolerance) * h_target,
                                 edgefuncs="Uniform",
                                 folder=folder_name_,
                                 plot_mesh=False,
                                 plot_sdf=False,
                                 h_func=h_func,
                                 file_name="mesh")


    uniform_box_mesh.mesh_plot(P, T, "mesh", folder=folder_name_,
                               save=True, L_R=2000, zoom=False,
                               aspect="equal", linewidth=1)
    
    
    dir_assurer("Barotropic FEM Objects")
    fem_dir = f"{folder_name_}/fem_N={order}.pkl"

    X, Y = P.T

    if not file_exist(fem_dir):
        with open(fem_dir, "wb") as outp:
            if verbose:
                print("Creating FEM class")
    
            fem = FEM(P, T, N=order)
            pickle.dump(fem, outp, pickle.HIGHEST_PROTOCOL)
    
    else:
        if verbose:
            print("Loading FEM class")
    
        with open(fem_dir, "rb") as inp:
            fem = pickle.load(inp)
            
    from barotropicSWEs import Barotropic
    from ppp.Plots import plot_setup, save_plot

    fig_vals_full, ax_vals_full = plot_setup("Re($\\omega$)", "Im($\\omega$)")

    fig_vals, ax_vals = plot_setup("Re($\\omega$)", "Im($\\omega$)")

    fig_err, ax_err = plot_setup("Eigenvalue (m,n)",
                                 "Absolute Error",
                                 y_log=True,
                                 by=error_base)
    
    for flux in numerical_fluxes:
        print(order, flux, h_target)
        swes = Barotropic.solver(
            fem,
            param,
            flux_scheme=flux,
            θ=theta_value,
            boundary_conditions="SOLID WALL",
            rotation=False
            )
        vals, vecs = swes.eigenvalue_problem()
        ax_vals_full.plot(vals.real, vals.imag, "x", label=flux)

        eigvals = np.zeros((4, 4), dtype=complex)
        analytical_vals = np.zeros((4, 4))
        errors = np.zeros((4, 4))
        x_ticks = []
        for i in range(4):
            for j in range(4):
                x_ticks.append(f"({i},{j})")
                analytical_vals[i, j] = np.pi * np.sqrt(i**2 + j**2)
                error = np.abs(vals - analytical_vals[i, j])
                index = np.argmin(error)
                errors[i, j] = error[index]
                eigvals[i, j] = vals[index]
                vals = np.delete(vals, index)
        
        ax_err.plot(errors.real.flatten("F")[1:],
                    "x-",
                    label=flux)

        ax_vals.plot(eigvals.real.flatten("F")[1:],
                     eigvals.imag.flatten("F")[1:],
                     "x",
                     label=flux)
        
    ax_vals.plot(analytical_vals.real.flatten("F")[1:],
                 analytical_vals.imag.flatten("F")[1:],
                 "ko",
                 mfc='none',
                 label="Analytical")
    
    analytical_vals = np.concatenate([-analytical_vals.flatten("F")[1:],
                                       analytical_vals.flatten("F")[1:]])
    ax_vals_full.plot(analytical_vals.real,
                      analytical_vals.imag,
                      "ko",
                      mfc='none',
                      label="Analytical")
    
    dir_assurer(f"{folder_name_}/Figures")

    ax_err.set_xticks(list(range(15)))
    ax_err.set_xticklabels(x_ticks[1:])
    save_plot(fig_err, ax_err, f'order={order}_Errors',
              folder_name=f"{folder_name_}/Figures")

    ax_vals.set_ylim([-tolerance, tolerance])
    save_plot(fig_vals, ax_vals, f'order={order}',
              folder_name=f"{folder_name_}/Figures")

    save_plot(fig_vals_full, ax_vals_full, f'order={order}_full',
              folder_name=f"{folder_name_}/Figures")
    
    ax_vals_full.set_xlim([-10, 10])
    ax_vals_full.set_ylim([-tolerance, tolerance])
    save_plot(fig_vals_full, ax_vals_full, f'order={order}_zoom',
              folder_name=f"{folder_name_}/Figures")
    
if __name__ == '__main__':
    from barotropicSWEs.Configuration import configure
    param = configure.main()
    fluxes = ["Central",
              "Upwind",
              "Penalty",
              "Lax-Friedrichs",
              "Alternating"
              ]

    square_eigenvalue_problem(param,
                              order=2,
                              h_target=.25,
                              tolerance=.1,
                              numerical_fluxes=fluxes,
                              theta_value=.25,
                              verbose=False
                              )