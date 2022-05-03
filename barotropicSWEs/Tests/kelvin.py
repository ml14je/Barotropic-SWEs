#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Fri Feb 25 11:59:04 2022

"""
import numpy as np

def kelvin_boundaryvalue_problem(param,
                                 bbox,
                                 order=2,
                                 h_target=.1,
                                 tolerance=.1,
                                 slope_profile='cos_squared',
                                 numerical_fluxes='Central',
                                 theta_value=.5,
                                 error_base=10,
                                 plot_topography=True,
                                 plot_exact=True,
                                 plot_numerical_flux_analysis=False,
                                 plot_spatial_error=False,
                                 regular_solutions=True,
                                 output_order=9,
                                 verbose=True
                                 ):
    from ppp.File_Management import dir_assurer, file_exist
    import pickle
    from DGFEM.dgfem import FEM
    from ppp.Plots import plot_setup, save_plot
    
    x0, xN, y0, yN = bbox 
    # (param.L_C+param.L_S)/param.L_R

    folder_name = f'BoundaryValue_Problem/h={h_target:.2e}/\
Domain={yN*1e-3*param.L_R:.0f}km/Slope={slope_profile}'
    dir_assurer(folder_name)
    file_name = f'Topography_shelfwidth={param.L_C*1e-3:.0f}km_\
slopewidth={param.L_S*1e-3:.0f}km_shelfdepth={param.H_C:.0f}m_\
abyssdepth={param.H_D:.0f}m'
    
    assert slope_profile.upper() in ['COS_SQUARED', 'GG07', 'LINEAR'], \
        'Invalid choice of slope profile - must be either COS_SQUARED, GG07 \
or LINEAR'

    from barotropicSWEs.Configuration.topography import coastal_topography        
    slope_topography, canyon_topography = coastal_topography(
        param,
        slope_choice=slope_profile.upper(),
        shelf_depth=param.H_C/param.H_D,
        coastal_shelf_width=param.L_C/param.L_R,
        coastal_lengthscale=(param.L_C+param.L_S)/param.L_R,
        smooth=False
        )

    if plot_topography:
        import matplotlib.pyplot as pt
        dir_assurer(f'{folder_name}')
        y = np.linspace(0, (param.L_C+2*param.L_S)/param.L_R, 1001)
        h = param.H_D * slope_topography(y)
        
        
        fig, ax = plot_setup('Cross-shore (km)',
                             'Coastal Bathymetry (m)')
        ax.plot(param.L_R * y * 1e-3, -h)
        pt.show()
        
        save_plot(fig, ax,
                  file_name,
                  folder_name=f'{folder_name}')
        
    ##### Kelvin Solution #####
    from barotropicSWEs.SWEs import kelvin_solver
    u_kelv, v_kelv, η_kelv, ω, k = kelvin_solver(
        slope_topography,
        param,
        coastal_lengthscale=(param.L_C + param.L_S)/param.L_R,
        tidal_amplitude=1,
        forcing_frequency=param.ω/param.f,
        foldername=f'{folder_name}/Kelvin Flow/{file_name}',
        filename=file_name)
    
    if plot_exact:
        x, y = np.linspace(x0, xN, 1001), np.linspace(y0, yN, 1001)
        Xg, Yg = np.meshgrid(x, y)
        for func, quantity, scale in zip([u_kelv, v_kelv, η_kelv],
                                         ['Alongshore_velocity',
                                          'Crossshore_velocity',
                                          'Surface_displacement'],
                                         [100 * param.c, 100 * param.c, param.ρ_ref * param.c ** 2]):
            fig, ax = plot_setup('Along-shore (km)',
                                 'Cross-shore (km)')
            vals = scale * func(Xg, Yg, 0)
            magnitude = np.max(np.abs(vals))
            c = ax.imshow(vals.real,
                          cmap="seismic",
                          extent=[L * param.L_R * 1e-3 for L in bbox],
                          vmin=-magnitude,
                          vmax=magnitude,
                          aspect='equal',
                          origin="lower")
            cbar = fig.colorbar(c, ax=ax)
            cbar.ax.tick_params(labelsize=16)
            save_plot(fig, ax,
                      f'{quantity}_DomainSize={yN*param.L_R*1e-3:.0f}km',
                      folder_name=f'{folder_name}/Kelvin Flow/{file_name}')
    
    if verbose:
        print(f'Associated Kelvin along-shore wavelength is \
{1e-3 * param.L_R * (2*np.pi/k):.2f} km')
    
    from barotropicSWEs.MeshGeneration import uniform_box_mesh
    P, T = uniform_box_mesh.main(bbox,
                                 (1-tolerance) * h_target,
                                 (1+tolerance) * h_target,
                                 edgefuncs="Uniform",
                                 folder=folder_name,
                                 plot_mesh=False,
                                 plot_sdf=False,
                                 h_func= lambda x, y : slope_topography(y),
                                 file_name="mesh")

    uniform_box_mesh.mesh_plot(P, T, "mesh", folder=folder_name,
                               save=True, L_R=2000, zoom=False,
                               aspect="equal", linewidth=1)

    dir_assurer(f"{folder_name}/FEM Objects")
    fem_dir = f"{folder_name}/FEM Objects/Kelvin_N={order}_{param.bbox}.pkl"

    X, Y = P.T
    wall_inds = np.where((Y == y0))[0]
    open_inds2 = np.where((X == x0) | (X == xN))[0]
    open_inds = np.where(Y == yN)[0]

    BC_maps, BC_Types = [wall_inds, open_inds, open_inds2], ["Wall", "Open", "Open2"]
    BCs = dict(zip(BC_Types, BC_maps))

    if not file_exist(fem_dir):
        with open(fem_dir, "wb") as outp:
            if verbose:
                print("Creating FEM class")

            fem = FEM(P, T, N=order, BCs=BCs)
            pickle.dump(fem, outp, pickle.HIGHEST_PROTOCOL)

    else:
        if verbose:
            print("Loading FEM class")

        with open(fem_dir, "rb") as inp:
            fem = pickle.load(inp)

    X, Y = np.round(fem.x.T.flatten(), 16), np.round(fem.y.T.flatten(), 16)
    flow_bg = (
        u_kelv(X, np.round(Y, 10), 0),
        v_kelv(X, np.round(Y, 10), 0),
    )
    h_func2 = np.vectorize(lambda x, y : slope_topography(y))
    
    from barotropicSWEs import Barotropic
    from barotropicSWEs import SWEs
    import time
    
    exact_sol = np.concatenate([
        u_kelv(X, np.round(Y, 10), 0),
        v_kelv(X, np.round(Y, 10), 0),
        η_kelv(X, np.round(Y, 10), 0)
        ], axis=0)[:, None]
    
    times = np.empty(len(numerical_fluxes))
    error_norms = np.empty((len(numerical_fluxes), 3))

    from ppp.P_Norms import p1_norm, p2_norm, pInf_norm
    v1 = np.ones((len(X), 1))
    norms = [p1_norm(v1), p2_norm(v1), pInf_norm(v1)]

    xg, yg = np.linspace(x0, xN, 1001), np.linspace(y0, yN, 1001)
    Xg, Yg = np.meshgrid(xg, yg)

    kelvin_numerical_solutions = {}
    for i, flux in enumerate(numerical_fluxes):
        start_time = time.perf_counter()

        if verbose:
            print(flux, order, h_target)

        swes = Barotropic.solver(
            fem,
            param,
            flux_scheme=flux,
            θ=theta_value,
            boundary_conditions="SPECIFIED",
            rotation=True,
            background_flow=flow_bg,
            h_func=h_func2,
            wave_frequency=ω,
            rayleigh_friction=0,
        )
        numerical_sol = SWEs.boundary_value_problem(
            swes,
            np.zeros(X.shape),
            animate=False,
            file_name=f'{flux.upper()}_order={order}',
            file_dir=f'{folder_name}/Irregular Solutions'
            )[:, None]
        
        times[i] = time.perf_counter() - start_time
        
        
        u_irreg, v_irreg, p_irreg = np.split(numerical_sol, 3)
        kelvin_solutions_irregular_grid = [u_irreg, v_irreg, p_irreg]

        if not regular_solutions:
            if order != output_order:
                # fem.PlotField2D(order,
                #                 param.L_R * 1e-3 * X,
                #                 param.L_R * 1e-3 * Y,
                #                 param.c * u_irreg.real,
                #                 title='Along-shore Velocity (m/s)',
                #                 x_label='ALong-shore (km)',
                #                 y_label='Cross-shore (km)',
                #                 name=None)

                # project velocity onto linear basis functions
                TRI, xout, yout, u_new, interp = fem.FormatData2D(
                    output_order, X, Y, u_irreg
                    )
                u_irreg = u_new
                v_irreg = interp @ v_irreg
                p_irreg = interp @ p_irreg
                kelvin_solutions_irregular_grid = [u_irreg, v_irreg, p_irreg]

            kelvin_numerical_solutions[flux] = kelvin_solutions_irregular_grid
            
        
        else:
            if not file_exist(
                    f'{flux.upper()}_order={order}.npz',
                    f'{folder_name}/Regular Solutions'
                    ):
                from ppp.Numpy_Data import save_arrays
                kelvin_solutions_regular_grid = []
                from scipy.interpolate import griddata
        
                for numerical_solution in kelvin_solutions_irregular_grid:
                    kelvin_solutions_regular_grid.append(
                        griddata(
                            (X, Y),
                            numerical_solution,
                            (Xg, Yg),
                            method="cubic")[:, :, 0]
                        )
                save_arrays(f'{flux.upper()}_order={order}',
                            kelvin_solutions_regular_grid,
                            folder_name=f'{folder_name}/Regular Solutions')
                
            else:
                from ppp.Numpy_Data import load_arrays
                u_reg, v_reg, p_reg = load_arrays(
                    f'{flux.upper()}_order={order}',
                    folder_name=f'{folder_name}/Regular Solutions'
                    )

                kelvin_solutions_regular_grid = [u_reg, v_reg, p_reg]

            kelvin_numerical_solutions[flux] = kelvin_solutions_regular_grid

        if plot_spatial_error:
            from scipy.interpolate import griddata
            import matplotlib.pyplot as pt
            
            v_shooting = param.c * v_kelv(Xg, np.round(Yg, 10), 0)
            v_dgfem = griddata((X, Y), param.c * v_irreg[:, 0],
                                (Xg, Yg),
                                method="cubic")
            
            fig_spatial_err, ax_spatial_err = plot_setup(
                "Along-shore (km)", "Cross-shore (km)")
            c = ax_spatial_err.imshow(
                np.log10(abs(v_shooting - v_dgfem)),
                cmap='plasma', origin='lower',
                extent=[x * 1e-3 * param.L_R for x in bbox],
                vmin=-16, vmax=0)
            
            for L in [param.L_C, param.L_C+param.L_S]:
                ax_spatial_err.hlines(L*1e-3,
                                      xmin=1e-3 * param.L_R * x0,
                                      xmax=1e-3 * param.L_R * xN,
                                      linestyle=':', color='k')
    
            cbar = fig_spatial_err.colorbar(c, ax=ax_spatial_err)
            cbar.set_ticks(np.linspace(-16, 0, 9, dtype=int))
            cbar.set_ticklabels([f'$10^{{{i}}}$' for i in \
                                  np.linspace(-16, 0, 9, dtype=int)])
            cbar.ax.tick_params(labelsize=16)
            save_plot(fig_spatial_err, ax_spatial_err,
                      f'{flux}_order={order}',
                      folder_name=f'{folder_name}/CrossShore_Error/Global')
            

            fig, ax = plot_setup("Along-shore (km)",
                                  "Cross-shore (km)")
            v_max = np.nanmax(np.abs(v_dgfem))
            c = ax.imshow(
                v_dgfem.real,
                cmap='seismic', origin='lower',
                extent=[x * 1e-3 * param.L_R for x in bbox],
                vmin=-v_max, vmax=v_max)
            
            for L in [param.L_C, param.L_C+param.L_S]:
                ax.hlines(L*1e-3,
                          xmin=1e-3 * param.L_R * x0,
                          xmax=1e-3 * param.L_R * xN,
                          linestyle=':', color='k')
    
            cbar = fig.colorbar(c, ax=ax)
            cbar.ax.tick_params(labelsize=16)
            save_plot(fig, ax,
                      f'{flux}_order={order}',
                      folder_name=f'{folder_name}/CrossShore_Error/Solution')
            
            
            for ind in [0, 500, 1000]:
                fig, ax = plot_setup('Cross-shore (km)',
                                     'Cross-shore velocity (cm/s)',
                                     scale=.6)
        
                ax.plot(param.L_R * 1e-3 * Yg[:, ind],
                        100*v_shooting[:, ind].real,
                        'r-', label='Shooting')
                ax.plot(param.L_R * 1e-3 * Yg[:, ind],
                        100*v_dgfem[:, ind].real,
                        'b-', label='DG-FEM')
                save_plot(fig, ax,
                          f'{flux}_order={order}',
                          my_loc=4,
                          folder_name=f'{folder_name}/CrossShore_Error/Slice_\
x={param.L_R * 1e-3 * Xg[0, ind]:.0f}km')
            
            fig, ax = plot_setup('Cross-shore (km)',
                                  'Relative Error')
            rel_err = np.abs(v_shooting.real - v_dgfem.real)/np.abs(v_shooting.real)
            for ind in [0, -1]:
                rel_err = np.abs(v_shooting.real[5:, ind] - \
                                  v_dgfem.real[5:, ind])/\
                    np.abs(v_shooting.real[5:, ind])
                ax.plot(param.L_R * 1e-3 * Yg[5:, ind],
                        rel_err,
                        label=f'$x={Xg[0, ind] * param.L_R * 1e-3 :.0f}$ km')
    
            ax.legend(fontsize=16, loc=1)
            save_plot(fig, ax,
                      f'{flux}_order={order}',
                      my_loc=1,
                      folder_name=f'{folder_name}/CrossShore_Error/Relative _Error')
        
        difference = numerical_sol - exact_sol
        
        for j, norm in enumerate([p1_norm, p2_norm, pInf_norm]):
            error_norms[i, j] = norm(difference)/norms[j]

    # raise ValueError
            
    if file_exist(f'order={order}.npz',
                  folder_name=f'{folder_name}/Time'):
        from ppp.Numpy_Data import load_arrays
        times, = load_arrays(f'order={order}',
                             folder_name=f'{folder_name}/Time')
        
    else:
        from ppp.Numpy_Data import save_arrays
        save_arrays(f'order={order}', [times],
                    folder_name=f'{folder_name}/Time')
            
    if plot_numerical_flux_analysis:
        import matplotlib.pyplot as pt
        fig_time, ax_time = plot_setup('Numerical Flux', 'Time (s)')
        ax_time.plot(range(len(numerical_fluxes)), times)
        ax_time.set_xticks(range(len(numerical_fluxes)))
        ax_time.set_xticklabels(numerical_fluxes)
        pt.show()
        
        fig_errors, ax_errors = plot_setup('Numerical Flux', 'Error', y_log=True,
                                           by=error_base)
        ax_errors.plot(range(len(numerical_fluxes)), error_norms.real, 'x-')
        ax_errors.set_xticks(range(len(numerical_fluxes)))
        ax_errors.set_xticklabels(numerical_fluxes)
        ax_errors.legend(['P1', 'P2', 'Pinf'], fontsize=16)
        pt.show()

    return times, error_norms

if __name__ == '__main__':
    from barotropicSWEs.Configuration import configure
    param = configure.main()
    fluxes = ["Central",
              "Upwind",
              "Penalty",
              "Lax-Friedrichs",
              "Alternating"
              ]

    kelvin_boundaryvalue_problem(param,
                                 param.bbox,
                                 order=2,
                                 h_target=1e-3,
                                 tolerance=.1,
                                 slope_profile='gg07',
                                 numerical_fluxes=fluxes,
                                 theta_value=.5,
                                 error_base=10,
                                 verbose=False
                                 )