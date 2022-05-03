#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Fri Feb 25 11:57:05 2022

"""
import numpy as np

def circle_eigenvalue_problem(param,
                              order=1,
                              radius=1.5,
                              h_target=.2,
                              epsilon=1e-8,
                              numerical_fluxes='Central',
                              theta_value=.5,
                              verbose=False
                              ):
    from ppp.File_Management import dir_assurer, file_exist
    import pickle
    from DGFEM.dgfem import FEM
    print('hi')
    
    if type(numerical_fluxes)==str:
        numerical_fluxes = [numerical_fluxes]
        
    R = radius
    folder_name_ = 'Eigenvalue_Problem/Rotating'
    folder_name2_ = f'{folder_name_}/h={h_target:.2f}'
    dir_assurer(folder_name_)
    
    if not file_exist(f"{folder_name2_}/mesh.npz"):
        from ppp.Numpy_Data import save_arrays
        import dmsh
        geo = dmsh.Circle([0, 0], radius)
        P, T = dmsh.generate(geo, h_target)
        save_arrays("mesh", [P, T],
                    folder_name=folder_name2_)
    else:
        from ppp.Numpy_Data import load_arrays
        P, T = load_arrays("mesh", folder_name=folder_name2_)
    
    from barotropicSWEs.MeshGeneration import uniform_box_mesh
    uniform_box_mesh.mesh_plot(P, T, "mesh", folder=folder_name2_,
                               save=True, L_R=param.L_R*1e-3, zoom=False,
                               aspect="equal", linewidth=1)
    
    # raise ValueError
    
    dir_assurer(f"{folder_name2_}/Barotropic FEM Objects")
    fem_dir = f"{folder_name2_}/Barotropic FEM Objects/fem_N={order}.pkl"

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
            
    if not file_exist(f"{folder_name_}/fundamental_frequencies_R={R:.2e}.npz"):
        from scipy.special import jv, jvp
        eigenvalues = np.array([])
        
        from ppp.Newton_Raphson import newton_raphson
        for n in range(7):
            kappa = lambda ω : np.sqrt(ω**2 - 1+0j)
            dispersion_relation = lambda ω : R * ω * kappa(ω) * jvp(n, R*kappa(ω)) - \
                                        n * jv(n, R*kappa(ω))
            
    
            for ω0 in np.linspace(-10, 10, 51):
                ω_return = newton_raphson(dispersion_relation, ω0,
                                          tolerance=1e-10)
                
                if ω_return is not None and abs(ω_return*(ω_return**2-1)) > 5e-2:
                    if len(eigenvalues) > 0:
                        if not np.any(abs(eigenvalues - ω_return) < 5e-2):
                            eigenvalues = np.append(eigenvalues, ω_return.real)
                            
                    else:
                        eigenvalues = np.append(eigenvalues, ω_return.real)
                        
        eigenvalues = np.sort(eigenvalues[abs(eigenvalues) < 10])
        analytical_vals = eigenvalues[np.argsort(np.abs(eigenvalues))]

        from ppp.Numpy_Data import save_arrays
        save_arrays(f"fundamental_frequencies_R={R:.2e}",
                    [analytical_vals],
                    folder_name=folder_name_)
        
    else:
        from ppp.Numpy_Data import load_arrays
        analytical_vals, = load_arrays(f"fundamental_frequencies_R={R:.2e}",
                                  folder_name=folder_name_)
    
    from ppp.Plots import plot_setup, save_plot
    from barotropicSWEs import Barotropic

    fig_vals_full, ax_vals_full = plot_setup("Re($\\omega$)", "Im($\\omega$)")

    fig_vals, ax_vals = plot_setup("Re($\\omega$)", "Im($\\omega$)")

    fig_err, ax_err = plot_setup("Eigenvalue",
                                 "Absolute Error",
                                 y_log=True)

    for flux in numerical_fluxes:
        print(flux, order)
        file_dir = f'{folder_name2_}/NumericalEigenvalues/Order={order}'
        dir_assurer(file_dir)
        print(f"{file_dir}/{flux.upper()}.npz", file_exist(f'{file_dir}/{flux.upper()}.npz'))
        if not file_exist(f'{file_dir}/{flux.upper()}.npz'):
            from ppp.Numpy_Data import save_arrays
            swes = Barotropic.solver(
                fem,
                param,
                flux_scheme=flux,
                θ=theta_value,
                boundary_conditions="SOLID WALL",
                rotation=True
                )
            vals, vecs = swes.eigenvalue_problem()
            save_arrays(f'{flux.upper()}', (vals,), folder_name=file_dir)
            
        else:
            from ppp.Numpy_Data import load_arrays
            vals, = load_arrays(f'{flux.upper()}', folder_name=file_dir)

        ax_vals_full.plot(vals.real, vals.imag, "x", label=flux)

        associated_vals = np.zeros(len(analytical_vals), dtype=complex)
        errors = np.zeros(len(analytical_vals))

        for i, val in enumerate(analytical_vals):
            error = np.abs(vals - val)
            index = np.argmin(error)
            associated_vals[i] = vals[index]
            errors[i] = error[index]
            vals = np.delete(vals, index)
        
        ax_err.plot(errors.real[1:20],
                    "x-",
                    label=flux)

        ax_vals.plot(associated_vals.real[1:],
                     associated_vals.imag[1:],
                     "x",
                     label=flux)
        
    ax_vals.plot(analytical_vals.real[1:],
                 analytical_vals.imag[1:],
                 "ko",
                 mfc='none',
                 label="Analytical")
    
    ax_vals_full.plot(analytical_vals.real[1:],
                      analytical_vals.imag[1:],
                      "ko",
                      mfc='none',
                      label="Analytical")
    
    dir_assurer(f"{folder_name_}/Figures")

    ax_err.set_xticks(list(range(19)))
    ax_err.set_xticklabels(list(range(1, 20)))
    save_plot(fig_err, ax_err, f'order={order}_Errors',
              folder_name=f"{folder_name_}/Figures")
    print(folder_name2_)
    save_plot(fig_err, ax_err, f'order={order}_Errors',
              folder_name=f"{folder_name2_}")

    ax_vals.set_ylim([-.1, .1])
    save_plot(fig_vals, ax_vals, f'order={order}',
              folder_name=f"{folder_name_}/Figures")

    save_plot(fig_vals_full, ax_vals_full, f'order={order}_full',
              folder_name=f"{folder_name_}/Figures")
    
    ax_vals_full.set_xlim([-10, 10])
    ax_vals_full.set_ylim([-.1, .1])
    save_plot(fig_vals_full, ax_vals_full, f'order={order}_zoom',
              folder_name=f"{folder_name_}/Figures")
    
    return errors


if __name__ == '__main__':
    from barotropicSWEs.Configuration import configure
    param = configure.main()
    fluxes = ["Central",
              "Upwind",
              "Penalty",
              "Lax-Friedrichs",
              "Alternating"
              ]
    
    circle_eigenvalue_problem(param,
                              order=2,
                              h_target=.25,
                              numerical_fluxes=fluxes,
                              theta_value=.25,
                              verbose=False
                              )