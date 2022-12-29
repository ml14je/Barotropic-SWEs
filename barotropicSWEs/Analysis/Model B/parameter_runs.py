#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Wed Mar 16 13:06:42 2022

"""
import numpy as np


def main(
    param,
    h_min,
    h_max,
    order,
    mesh_name="",
    canyon_width=15,  # in kilometres
    alpha=.35, # ND
    beta=.5, #ND
    plot_domain=False,
    numerical_flux="Lax-Friedrichs",
    potential_forcing=False,
    max_tidal_amplitude=1,
    plot_slope_topography=False,
    plot_canyon_topography=False,
    plot_mesh=False,
    slope_solution=None,
    plot_solution=False,
    plot_pertubation=False,
    fine_mesh=True,
    Nres=1000,
    save_all=True,
    θ=0.5,
    rotation=True,
    verbose=True,
):
    param.alpha, param.beta, param.canyon_width = alpha, beta, canyon_width

    import pickle
    from DGFEM.dgfem import FEM
    from barotropicSWEs.Configuration.topography import grad_function
    from ppp.Plots import plot_setup, save_plot

    assert numerical_flux.upper() in [
        "CENTRAL",
        "UPWIND",
        "LAX-FRIEDRICHS",
        "RIEMANN",
        "PENALTY",
        "ALTERNATING",
    ], "Invalid choice of numerical flux."

    x0, xN, y0, yN = param.bbox_dimensional
    Lx, Ly = xN - x0, yN - y0
    from ppp.File_Management import file_exist, dir_assurer

    folder_name = "Parameter Sweeps"
    dir_assurer(folder_name)
    file_name = f"Domain={Lx:.0f}kmx{Ly:.0f}km_OceanDepth={param.H_D:.0f}m_\
ShelfDepth={param.H_C:.0f}m_\
ShelfWidth={param.L_C*1e-3:.0f}km_SlopeWidth={param.L_S*1e-3:.0f}km"

    # Non-Dimensional Shelf Parameters
    L0, λ = param.L_C / param.L_R, (param.L_C + param.L_S) / param.L_R
    # HC = param.H_C / param.H_D

    # Non-Dimensional Canyon Parameters
    W = canyon_width * 1e3 / param.L_R

    # Topography
    from barotropicSWEs.Configuration import topography_sdg_revision as topography

    slope_temp, canyon_temp = topography.coastal_topography(
        param
    )
    
    slope_topography = lambda y : slope_temp(y * param.L_R) / param.H_D
    canyon_topography = lambda x, y : \
        canyon_temp(x * param.L_R, y * param.L_R) / param.H_D

    if plot_slope_topography:
        y = np.linspace(0, (param.L_C + 2 * param.L_S) / param.L_R, 1001)
        h = param.H_D * slope_topography(y)

        fig, ax = plot_setup("Cross-shore (km)", "Coastal Bathymetry (m)")
        ax.plot(param.L_R * y * 1e-3, -h)
        save_plot(fig, ax, file_name, folder_name=folder_name)

    # Far field barotropic tide in form of perturbed Kelvin Wave
    from barotropicSWEs.SWEs import kelvin_solver

    u_kelv, v_kelv, p_kelv, ω, k = kelvin_solver(
        slope_topography,
        param,
        coastal_lengthscale=(param.L_C + param.L_S) / param.L_R,
        tidal_amplitude=1,
        forcing_frequency=param.ω / param.f,
        foldername=f"{folder_name}/Kelvin Flow/{file_name}",
        filename=f"KelvinMode_Domain={Lx}kmx{Ly}km_ω={param.ω:.1e}",
    )
    print(alpha, beta, canyon_width)
    if alpha < 1e-2 or beta < 1e-2 or canyon_width < 1:
        file_name += '_Slope'
    else:
        file_name += f"_CanyonWidth={canyon_width:.1f}km\
_Alpha={alpha:.2f}_Beta={beta:.2f}"

    h_func_dim = lambda x, y: param.H_D * canyon_topography(x, y)

    h_min = (λ - L0) / 10 if (W < 1e-4 or \
        beta < 1e-2 or \
            alpha < 1e-2) \
                else min(W / 4, (λ - L0) / 20)

    from barotropicSWEs.MeshGeneration import make_canyon_meshes

    P, T, mesh_name = make_canyon_meshes.mesh_generator(
        param.bbox,
        param,
        h_min,
        h_max / 3,
        canyon_topography,
        mesh_dir="Meshes",
        plot_mesh=False,
        plot_sdf=False,
        plot_edgefunc=False,
        max_iter=1000,
        mesh_gradation=0.35,
        slope_parameter=28,
        verbose=True,
        save_mesh=save_all,
        model='B',
    )

    if plot_mesh:
        from barotropicSWEs.MeshGeneration import uniform_box_mesh
        uniform_box_mesh.mesh_plot(
            P,
            T,
            file_name,
            folder="Meshes/Figures",
            h_func=h_func_dim,
            save=True,
            L_R=param.L_R * 1e-3,
            zoom=False,
        )

    file_name += f"_ForcingFrequency={param.ω:.1e}s^{{{-1}}}"
    dir_assurer("Barotropic FEM Objects")
    fem_dir = f"Barotropic FEM Objects/{mesh_name}_N={order}.pkl"

    # Non-dimensionalise bbox coordinates
    x0, xN, y0, yN = (
        x0 * 1e3 / param.L_R,
        xN * 1e3 / param.L_R,
        y0 * 1e3 / param.L_R,
        yN * 1e3 / param.L_R,
    )

    X, Y = P.T
    wall_inds = np.where((Y == y0))[0]
    open_inds2 = np.where((X == x0) | (X == xN))[0]
    open_inds = np.where(Y == yN)[0]

    BC_maps, BC_Types = [wall_inds, open_inds, open_inds2], \
        ["Wall", "Open", "Open2"]
    BCs = dict(zip(BC_Types, BC_maps))

    if not file_exist(fem_dir):
        if save_all:
            with open(fem_dir, "wb") as outp:
                if verbose:
                    print("Creating FEM class")
    
                fem = FEM(P, T, N=order, BCs=BCs)
                pickle.dump(fem, outp, pickle.HIGHEST_PROTOCOL)
                
        else:
            fem = FEM(P, T, N=order, BCs=BCs)   

    else:
        if verbose:
            print("Loading FEM class")

        with open(fem_dir, "rb") as inp:
            fem = pickle.load(inp)

    X, Y = np.round(fem.x.T.flatten(), 15), np.round(fem.y.T.flatten(), 15)

    background_flow = (
        u_kelv(X, Y, 0),
        v_kelv(X, Y, 0),
    )

    from barotropicSWEs import Barotropic

    swes = Barotropic.solver(
        fem,
        param,
        flux_scheme=numerical_flux,
        θ=θ,
        boundary_conditions="SPECIFIED",
        rotation=True,
        background_flow=background_flow,
        h_func=canyon_topography,
        wave_frequency=ω,
        rayleigh_friction=0,
    )

    from barotropicSWEs import SWEs

    numerical_sol = SWEs.boundary_value_problem(
        swes,
        np.zeros(X.shape),
        animate=False,
        file_name=f"{numerical_flux.upper()}_order={order}",
        file_dir=f"{folder_name}/Solutions/{file_name}",
        save_solution=save_all,
    )


    if plot_solution:
        from barotropicSWEs.Analyses.process_slns import plot_solutions
        plot_solutions(
            param,
            swes,
            numerical_sol,
            zoom=False,
            bbox_plot=param.bbox_dimensional,
            file_dir=f"{folder_name}/Figures/{file_name}",
        )
        return None

    del swes, fem

    u_num, v_num, p_num = np.split(numerical_sol, 3)

    assert len(u_num) == len(X)
    canyon_irreg = np.array([u_num, v_num, p_num])
    
    
    if slope_solution is None:
        return (X, Y, canyon_irreg)
    
    else:
        X_slope, Y_slope, slope_irreg = slope_solution
        
    if fine_mesh:
        y00 = 0
        Lx, Ly = 30e3/param.L_R, 200e3/param.L_R
        res_dim = (.125 * 1e3, .125 * 1e3) #in metres
        aspect_ = 'auto'

    else:
        y00 = 60 * 1e3/param.L_R
        Lx, Ly = (xN - x0)/2, (yN - y0)/2
        res_dim = (param.L_R * Lx / Nres,
                   param.L_R * Ly / Nres) 
        aspect_ = "equal"

    res = (res_dim[0]/param.L_R, res_dim[1]/param.L_R)
    x = np.arange(-Lx/2, Lx/2 + res[0], res[0])
    y = np.arange(y00, y00 + Ly + res[1], res[1])
    Xg, Yg = np.meshgrid(x, y)
    Xg = np.round(Xg, 15)
    Yg = np.round(Yg, 15)
    val_dict = {}

    bbox_dim = (-Lx * 1e-3 * param.L_R/2,
                 Lx * 1e-3 * param.L_R/2,
                 y00 * 1e-3 * param.L_R,
                 (y00 + Ly) * 1e-3 * param.L_R)

    from scipy.interpolate import griddata
    canyon_reg_sol, slope_reg_sol = [], []

    scales = [100 * param.c, 100 * param.c ,
              param.ρ_min * (param.c ** 2)]

    val_dict['magnitude of change'] = {'u' : {}, 'v': {}, 'p' : {}, 'w_x' : {},
                                       'w_y' : {}, 'w' : {}}
    val_dict['change of magnitude'] = {'u' : {}, 'v': {}, 'p' : {}, 'w_x' : {},
                                       'w_y' : {}, 'w' : {}}
    val_dict['solution'] = {'u' : {}, 'v': {}, 'p' : {}, 'w_x' : {},
                            'w_y' : {}, 'w' : {}, 'h_x' : {}, 'h_y' : {}, 'grad_h' : {}}
    Xg_flatten, Yg_flatten = Xg.flatten(), Yg.flatten()
    
    for i, key in enumerate(['u', 'v', 'p']):
        canyon_reg_sol.append(griddata((X, Y),
                                       scales[i] * canyon_irreg[i],
                                       (Xg, Yg),
                                       method="cubic"))
        slope_reg_sol.append(griddata((X_slope, Y_slope),
                                      scales[i] * slope_irreg[i],
                                      (Xg, Yg),
                                      method="cubic"))

        sln = np.abs(canyon_reg_sol[-1]).flatten()
        change_in_magnitude = np.abs(np.abs(canyon_reg_sol[-1]) - \
            np.abs(slope_reg_sol[-1])).flatten()
        magnitude_of_change = np.abs(canyon_reg_sol[-1] - \
                                     slope_reg_sol[-1]).flatten()
        ind1 = np.nanargmax(change_in_magnitude)
        ind2 = np.nanargmax(magnitude_of_change)
        ind3 = np.nanargmax(sln)
        val_dict['change of magnitude'][key]['x'] = Xg_flatten[ind1]
        val_dict['change of magnitude'][key]['y'] = Yg_flatten[ind1]
        val_dict['change of magnitude'][key]['Linf'] = change_in_magnitude[ind1]
        val_dict['change of magnitude'][key]['L1'] = np.mean(change_in_magnitude)
        val_dict['change of magnitude'][key]['L2'] = \
            np.sqrt(np.sum(change_in_magnitude ** 2))/np.sqrt(len(change_in_magnitude))
        
        val_dict['magnitude of change'][key]['x'] = Xg_flatten[ind2]
        val_dict['magnitude of change'][key]['y'] = Yg_flatten[ind2]
        val_dict['magnitude of change'][key]['Linf'] = magnitude_of_change[ind2]
        val_dict['magnitude of change'][key]['L1'] = np.mean(magnitude_of_change)
        val_dict['magnitude of change'][key]['L2'] = \
            np.sqrt(np.sum(magnitude_of_change ** 2))/np.sqrt(len(magnitude_of_change))
            
        val_dict['solution'][key]['x'] = Xg_flatten[ind3]
        val_dict['solution'][key]['y'] = Yg_flatten[ind3]
        val_dict['solution'][key]['Linf'] = sln[ind3]
        val_dict['solution'][key]['L1'] = np.mean(sln)
        val_dict['solution'][key]['L2'] = \
            np.sqrt(np.sum(sln ** 2))/np.sqrt(len(sln))
            
        
    
    canyon_depth = h_func_dim(Xg, Yg)
    slope_depth = param.H_D * slope_topography(Yg)
    canyon_h_x, canyon_h_y = grad_function(canyon_depth, res_dim[1], res_dim[0])
    gradh_canyon = np.sqrt(canyon_h_x**2 +  canyon_h_y**2)
    slope_h_x, slope_h_y = grad_function(slope_depth, res_dim[1], res_dim[0])    

    NNx, NNy = max(1, len(x)// 25), max(len(y)// 25, 1)
    u_canyon, v_canyon, p_canyon = canyon_reg_sol

    del canyon_reg_sol
    w_x_canyon, w_y_canyon = canyon_h_x * u_canyon, canyon_h_y * v_canyon
    w_canyon = w_x_canyon + w_y_canyon
    
    for gradh, key in zip([canyon_h_x, canyon_h_y, gradh_canyon],
                          ['h_x', 'h_y', 'grad_h']):
        sln = np.abs(gradh).flatten()
        ind = np.nanargmax(sln)
        val_dict['solution'][key]['x'] = Xg_flatten[ind]
        val_dict['solution'][key]['y'] = Yg_flatten[ind]
        val_dict['solution'][key]['Linf'] = sln[ind]
        val_dict['solution'][key]['L1'] = np.mean(sln)
        val_dict['solution'][key]['L2'] = \
            np.sqrt(np.sum(sln ** 2))/np.sqrt(len(sln))
    
    del canyon_h_x, canyon_h_y

    u_slope, v_slope, p_slope = slope_reg_sol

    del slope_reg_sol
    w_x_slope, w_y_slope = slope_h_x * u_slope, slope_h_y * v_slope
    w_slope = w_x_slope + w_y_slope
    del slope_depth, slope_h_x, slope_h_y

    w_perturbation = w_canyon - w_slope
    

    
    for w_s, w_c, key in zip([w_x_slope, w_y_slope, w_slope],
                             [w_x_canyon, w_y_canyon, w_canyon],
                             ['w_x', 'w_y', 'w']):
        sln = np.abs(w_c).flatten()
        change_in_magnitude = np.abs(np.abs(w_c) - np.abs(w_s)).flatten()
        magnitude_of_change = np.abs(w_c - w_s).flatten()
        ind1 = np.nanargmax(change_in_magnitude)
        ind2 = np.nanargmax(magnitude_of_change)
        ind3 = np.nanargmax(sln)
        val_dict['change of magnitude'][key]['x'] = Xg_flatten[ind1]
        val_dict['change of magnitude'][key]['y'] = Yg_flatten[ind1]
        val_dict['change of magnitude'][key]['Linf'] = change_in_magnitude[ind1]
        val_dict['change of magnitude'][key]['L1'] = np.mean(change_in_magnitude)
        val_dict['change of magnitude'][key]['L2'] = \
            np.sqrt(np.sum(change_in_magnitude ** 2))/np.sqrt(len(change_in_magnitude))
        
        val_dict['magnitude of change'][key]['x'] = Xg_flatten[ind2]
        val_dict['magnitude of change'][key]['y'] = Yg_flatten[ind2]
        val_dict['magnitude of change'][key]['Linf'] = magnitude_of_change[ind2]
        val_dict['magnitude of change'][key]['L1'] = np.mean(magnitude_of_change)
        val_dict['magnitude of change'][key]['L2'] = \
            np.sqrt(np.sum(magnitude_of_change ** 2))/np.sqrt(len(magnitude_of_change))
            
        val_dict['solution'][key]['x'] = Xg_flatten[ind3]
        val_dict['solution'][key]['y'] = Yg_flatten[ind3]
        val_dict['solution'][key]['Linf'] = sln[ind3]
        val_dict['solution'][key]['L1'] = np.mean(sln)
        val_dict['solution'][key]['L2'] = \
            np.sqrt(np.sum(sln ** 2))/np.sqrt(len(sln))

    del w_slope, w_canyon

    p_perturbation = p_canyon - p_slope
    del p_slope, p_canyon

    u_slope2 = u_slope[::NNy, ::NNx]
    del u_slope
    u_canyon2 = u_canyon[::NNy, ::NNx]
    del u_canyon
    v_slope2 = v_slope[::NNy, ::NNx]
    del v_slope
    v_canyon2 = v_canyon[::NNy, ::NNx]
    del v_canyon

    if plot_pertubation:
        for phase in np.arange(8) / 8:
            fig_vel, ax_vel = plot_setup(
                x_label="Along-shore (km)", y_label="Cross-shore (km)"
            )
            fig_p, ax_p = plot_setup(
                x_label="Along-shore (km)", y_label="Cross-shore (km)"
            )
            fig_w, ax_w = plot_setup(
                x_label="Along-shore (km)", y_label="Cross-shore (km)"
            )
            c = ax_p.imshow(
                (p_perturbation * np.exp(2j * np.pi * phase)).real,
                cmap="seismic",
                aspect=aspect_,
                origin="lower",
                vmin=-val_dict['magnitude of change']['p']['Linf'],
                vmax= val_dict['magnitude of change']['p']['Linf'],
                extent=bbox_dim,
            )
            cbar = fig_p.colorbar(c, ax=ax_p)
            cbar.ax.tick_params(labelsize=16)

            c = ax_w.imshow(
                (w_perturbation * np.exp(2j * np.pi * phase)).real,
                cmap="seismic",
                aspect=aspect_,
                origin="lower",
                vmin=-val_dict['magnitude of change']['w']['Linf'],
                vmax= val_dict['magnitude of change']['w']['Linf'],
                extent=bbox_dim,
            )
            cbar = fig_p.colorbar(c, ax=ax_w)
            cbar.ax.tick_params(labelsize=16)

            for ax in [ax_vel, ax_p, ax_w]:
                ax.contour(
                    Xg * 1e-3 * param.L_R,
                    Yg * 1e-3 * param.L_R,
                    canyon_depth,
                    sorted([param.H_C + 1e-3,
                            param.canyon_depth,
                            param.H_D - 1e-3]),
                    colors="black",
                )

            save_plot(
                fig_p,
                ax_p,
                f"Pressure_Perturbation_{phase}_fine={fine_mesh}",
                folder_name=f"{folder_name}/Figures/{file_name}/\
Flux={numerical_flux.upper()}_order={order}",
            )
            del fig_p, ax_p

            save_plot(
                fig_w,
                ax_w,
                f"Vertical_Velocity_Perturbation_{phase}_fine={fine_mesh}",
                folder_name=f"{folder_name}/Figures/{file_name}/\
Flux={numerical_flux.upper()}_order={order}",
            )
            del fig_w, ax_w

            for u, v, c in zip(
                [u_slope2, u_canyon2], [v_slope2, v_canyon2], ["b", "r"]
            ):
                Q = ax_vel.quiver(
                    1e-3 * param.L_R * Xg[::NNy, ::NNx],
                    1e-3 * param.L_R * Yg[::NNy, ::NNx],
                    (u * np.exp(2j * np.pi * phase)).real/100,
                    (v * np.exp(2j * np.pi * phase)).real/100,
                    width=0.002,
                    scale=1,
                    color=c,
                )

            ax_vel.quiverkey(
                Q,
                0.7,
                0.04,
                .05,
                r"$5\,\rm{cm/s}$",
                labelpos="W",
                coordinates="figure",
                fontproperties={"weight": "bold", "size": 18},
            )
            ax_vel.set_aspect(aspect_)
            save_plot(
                fig_vel,
                ax_vel,
                f"Velocity_Comparison_{phase}_fine={fine_mesh}",
                folder_name=f"{folder_name}/Figures/{file_name}/\
Flux={numerical_flux.upper()}_order={order}",
            )

            del fig_vel, ax_vel

        del (
            u_slope2,
            v_slope2,
            u_canyon2,
            v_canyon2,
            p_perturbation,
            w_perturbation,
            Xg,
            Yg,
        )

    return val_dict

def parameter_run(
    param,
    canyon_width,
    alpha_values,
    beta_values,
    order=3,
    numerical_flux="Central",
    slope_choice="Cos_Squared",
    h_min=5e-4,
    h_max=1e-2,
    plot_slope_topography=False,
    goal='Solutions'
):
    from ppp.File_Management import dir_assurer, file_exist
    import pickle

    assert goal.upper() in ['NORMS', 'SOLUTIONS', 'PERTURBATIONS'], \
        "Invalid choice of goal."
    alpha_save, beta_save = np.array([.8, .5, .35, .2]), np.array([0, .2, .5, .8])
          
    if goal.upper() == 'SOLUTIONS':
        fine_mesh_ = False
        plot_perturbation_ = False
        plot_solution_ = True
        
        alpha_values = alpha_save
        beta_values = beta_save
    
    elif goal.upper() == 'PERTURBATIONS':
        fine_mesh_ = False
        plot_perturbation_ = True
        plot_solution_ = False
        
        alpha_values = alpha_save
        beta_values = beta_save

    else:
        fine_mesh_ = True
        plot_perturbation_ = False
        plot_solution_ = False

    data_name = f"CanyonWidth={canyon_width:.1f}km\
_flux={numerical_flux}_order={order}"
    folder_name_ = "Parameter Sweeps/Data Tables"
    dir_assurer(folder_name_)
    slope_solution_ = main(
        param,
        h_min,
        h_max,
        order,
        mesh_name="",
        canyon_width=0,  # in kilometres
        alpha=0,  # ND
        beta=0,  # ND
        plot_domain=False,
        numerical_flux=numerical_flux,
        potential_forcing=False,
        max_tidal_amplitude=1,
        plot_slope_topography=False,
        plot_canyon_topography=False,
        θ=0.5,
        rotation=True,
        plot_solution=plot_solution_,
        plot_pertubation=plot_perturbation_,
        fine_mesh=fine_mesh_,
        verbose=True,
        slope_solution=None,
    )
    
    print(f"Width: {canyon_width:.1f} km", flush=True)
    
    if goal.upper() == 'NORMS':
        if not file_exist(f"{folder_name_}/{data_name}.pkl") or goal.upper() != 'NORMS':
            dictionaries = {}
            
         
        else:
            with open(f"{folder_name_}/{data_name}.pkl", "rb") as inp:
                dictionaries = pickle.load(inp)
                
                try:
                    dictionaries['slope']
                    
                except KeyError:
                    slope_dictionary = main(
                        param,
                        h_min,
                        h_max,
                        order,
                        mesh_name="",
                        canyon_width=0,  # in kilometres
                        alpha=0,  # ND
                        beta=0,  # ND
                        plot_domain=False,
                        numerical_flux=numerical_flux,
                        potential_forcing=False,
                        max_tidal_amplitude=1,
                        plot_slope_topography=False,
                        plot_canyon_topography=False,
                        θ=0.5,
                        rotation=True,
                        plot_solution=plot_solution_,
                        plot_pertubation=plot_perturbation_,
                        fine_mesh=fine_mesh_,
                        verbose=True,
                        slope_solution=slope_solution_,
                    )
                    dictionaries['slope'] = slope_dictionary
                    
                
        print(f"{100 * len(dictionaries.keys())/(49**2) :.2f}%", flush=True)

    for i, alpha in enumerate(alpha_values):
        flag = False
        for j, beta in enumerate(beta_values[1:]):
            param.canyon_depth = param.H_D - (param.H_D - param.H_C) * np.cos(beta * np.pi/2)**2
            param.alpha, param.beta = alpha, beta
            save_all = alpha in alpha_save and beta in beta_save
            # print(f'alpha: {alpha:.3f}, beta: {beta:.3f}, save: {save_all}', flush=True)
            
            if goal.upper() in ['SOLUTIONS', 'PERTURBATIONS']:
                main(
                    param,
                    h_min,
                    h_max,
                    order,
                    mesh_name="",
                    canyon_width=canyon_width,  # in kilometres
                    alpha=param.alpha,  # ND
                    beta=param.beta,  # ND
                    plot_domain=False,
                    numerical_flux=numerical_flux,
                    potential_forcing=False,
                    max_tidal_amplitude=1,
                    plot_slope_topography=False,
                    plot_canyon_topography=False,
                    θ=0.5,
                    rotation=True,
                    plot_solution=plot_solution_,
                    plot_pertubation=plot_perturbation_,
                    fine_mesh=fine_mesh_,
                    save_all=save_all,
                    verbose=True,
                    slope_solution=slope_solution_,
                )
            
            else:
                try:
                    dictionaries[(param.alpha, param.beta)]
                    
                except KeyError:
                    dictionary = main(
                        param,
                        h_min,
                        h_max,
                        order,
                        mesh_name="",
                        canyon_width=canyon_width,  # in kilometres
                        alpha=param.alpha,  # ND
                        beta=param.beta,  # ND
                        plot_domain=False,
                        numerical_flux=numerical_flux,
                        potential_forcing=False,
                        max_tidal_amplitude=1,
                        plot_slope_topography=False,
                        plot_canyon_topography=False,
                        θ=0.5,
                        rotation=True,
                        plot_solution=False,
                        plot_pertubation=False,
                        fine_mesh=fine_mesh_,
                        save_all=save_all,
                        verbose=True,
                        slope_solution=slope_solution_,
                    )
                
                    dictionaries[(param.alpha, param.beta)] = dictionary
                    flag = True
        
        if goal.upper() == 'NORMS' and flag:
            with open(f"{folder_name_}/{data_name}.pkl", "wb") as outp:
                pickle.dump(dictionaries, outp, pickle.HIGHEST_PROTOCOL)

    from ppp.Plots import plot_setup, save_plot
    import matplotlib.pyplot as pt
    
    ### Position of maximum ###
    # colors = ['b', 'r']
    # labels = ['Along-Shore', 'Cross-Shore']
    # labels2 = ['Along-shore velocity', 'Cross-shore velocity',
    #            'Pressure', 'Vertical velocity']

    # for ind, key in enumerate(['change of magnitude', 'magnitude of change']):
    #     for pos, label in zip(['x', 'y'], labels):
    #         fig, ax = plot_setup(x_label, f"{label} (km)", title=key)
    #         for ind2, key2 in enumerate(['u', 'v', 'p', 'w']):
    #             vals = [dictionaries[i][key][key2][pos] * 1e-3 * param.L_R for \
    #                     i in range(len(dictionaries))]
    #             ax.plot(parameter_values[1:], vals,
    #                     label=f'{labels2[ind2]}',
    #                     markersize=12, markerfacecolor='white')
                
    #         pt.legend(fontsize=18, loc=1)
    #         pt.show()

    if goal.upper() == 'NORMS':
        def plot_contours(ax):
            L_levels = [20, 60, 100, 140]
            H_levels = [400, 1000, 2000, 3000, 3800]
            def fmt_func(L):
                return f'{L:.0f} km'
            
            def fmt_func2(H):
                return f'{H:.0f} m'
                
            cs = ax.contour(A, B,
                            (A * param.L_C + B * param.L_S) * 1e-3,
                            L_levels, colors='k')
            ax.clabel(cs, cs.levels, inline=True,
                      fmt=fmt_func, fontsize=18)
            
            
            cs = ax.contour(A, B,
                            param.H_D - (param.H_D - param.H_C) * np.cos(B * np.pi/2)**2,
                            H_levels, colors='k')
            ax.clabel(cs, cs.levels, inline=True,
                      fmt=fmt_func2, fontsize=18)

        # ### Pressure ###
        key2 = 'p'
        v_mins = [9.45, 9.45, 9.85]
        v_maxs = [9.5, 9.5, 10.05]
        
        import matplotlib.ticker as tick
        from ppp.Plots import add_colorbar

        for ind, key in enumerate(['solution']):#'change of magnitude',
                                    # 'magnitude of change', ]):
            for i, norm in enumerate(['L1', 'L2', 'Linf']):
                fig, ax = plot_setup('$\\alpha$', '$\\beta$', scale=.7)#,
                                      # title="Pressure (Pa)")
                
                A, B = np.meshgrid(alpha_values, beta_values)
                vals = np.array([dictionaries[(alpha, beta)][key][key2][norm] * 1e-3 \
                                  for alpha, beta in \
                                      zip(A[1:].flatten(), B[1:].flatten())]).reshape(A[1:].shape)
                vals_slope = np.array([dictionaries['slope'][key][key2][norm] * 1e-3 \
                                  for alpha in alpha_values])[None, :]
                vals = np.concatenate([vals_slope, vals], axis=0) 
                c = ax.contourf(vals, origin='lower',
                              extent=[alpha_values[0], alpha_values[-1],
                                      beta_values[0], beta_values[-1]],
                              cmap='inferno_r',
                              levels=np.linspace(v_mins[i], v_maxs[i], 21),
                              extend='both'
                              )
                ax.set_aspect('equal')
                cbar = add_colorbar(c,
                                    format=tick.FormatStrFormatter('%.2f'))
                
                cbar.ax.set_ylabel('Pressure ($\\rm{kPa}$)', rotation=270,
                        fontsize=16, labelpad=20)
                cbar.ax.tick_params(labelsize=16)
                # plot_contours(ax)
                # ax.plot([.35], [.5], 'x', color='red', ms=10, mew=3, ls='none')
                # ax.plot(np.tile(alpha_save, len(beta_save)),
                #         np.repeat(beta_save, len(alpha_save)),
                #         'rx')
                # pt.show()
            
                save_plot(fig, ax, f"Pressure_CanyonWidth={canyon_width:.1f}km_{norm}",
                          my_loc=2,
                          folder_name=f"Parameter Sweeps/Figures/Parameters/{key.replace(' ', '_')}")
                

        # Horizontal Velocity ###
        labels = ['Along-shore', 'Cross-shore']
        v_mins = [(2.9, 2), (3.24, 2.5), (4.5, 3.75)]
        v_maxs = [(3.33, 2.62), (3.5, 3.4), (6.6, 10.1)]
        
        for key in ['solution']:#, 'change of magnitude', 'magnitude of change']:
            for i, norm in enumerate(['L1', 'L2', 'Linf']):
                for ind, key2 in enumerate(['u', 'v']):
                    fig, ax = plot_setup('$\\alpha$', '$\\beta$', scale=.7)#,
                                          # title=f"{labels[ind]} Speed (cm/s)")
                    A, B = np.meshgrid(alpha_values, beta_values)
                    vals = np.array([dictionaries[(alpha, beta)][key][key2][norm] \
                                      for alpha, beta in \
                                          zip(A[1:].flatten(), B[1:].flatten())]).reshape(A[1:].shape)
                    vals_slope = np.array([dictionaries['slope'][key][key2][norm] \
                                      for alpha in alpha_values])[None, :]
                    vals = np.concatenate([vals_slope, vals], axis=0)
                    c = ax.contourf(vals, origin='lower',
                                  extent=[alpha_values[0], alpha_values[-1],
                                          beta_values[0], beta_values[-1]],
                                  cmap='inferno_r',
                                  levels=np.linspace(v_mins[i][ind],v_maxs[i][ind], 21),
                                  extend='both'
                                  )
                    ax.set_aspect('equal')
                    cbar = add_colorbar(c, ax=ax,
                                        format=tick.FormatStrFormatter('%.2f'))
                    cbar.ax.set_ylabel(f'{labels[ind]} velocity ($\\rm{{cm/s}}$)', rotation=270,
                            fontsize=16, labelpad=20)
                    cbar.ax.tick_params(labelsize=16)
                    # plot_contours(ax)
                    # ax.plot([.35], [.5], 'x', color='red', ms=10, mew=3, ls='none')
                    # ax.plot(np.tile(alpha_save, len(beta_save)),
                    #         np.repeat(beta_save, len(alpha_save)),
                    #         'rx')
                    # pt.show()
                
                    save_plot(fig, ax, f"{labels[ind]}Velocity_CanyonWidth={canyon_width:.1f}km_{norm}",
                              my_loc=2,
                              folder_name=f"Parameter Sweeps/Figures/Parameters/{key.replace(' ', '_')}")

        # # Bathymetric Gradient ###
        # labels = ['Along-shore', 'Cross-shore', 'Absolute']
        # v_mins = [0, 0]
        # v_maxs = [0.1, 2]

        # for ind, key2 in enumerate(['h_x', 'grad_h']):
        #     for i, norm in enumerate(['L1', 'Linf']):
        #         fig, ax = plot_setup('$\\alpha$', '$\\beta$', scale=.7)#,
        #                               # title="Bathymetric Gradient")
        #         print(alpha_values, beta_values)
        #         A, B = np.meshgrid(alpha_values, beta_values)
        #         vals = np.array([dictionaries[(alpha, beta)]['solution'][key2][norm] \
        #                           for alpha, beta in \
        #                               zip(A[1:].flatten(), B[1:].flatten())]).reshape(A[1:].shape)
        #         vals_slope = np.array([dictionaries['slope']['solution'][key2]['Linf'] \
        #                           for alpha in alpha_values])[None, :]
        #         vals = np.concatenate([vals_slope, vals], axis=0)
        #         c = ax.contourf(vals, aspect='equal', origin='lower',
        #                       extent=[alpha_values[0], alpha_values[-1],
        #                               beta_values[0], beta_values[-1]],
        #                       cmap='inferno_r',
        #                       levels=np.linspace(v_mins[i], v_maxs[i], 21)
        #                       )
        #         cbar = add_colorbar(c, ax=ax)
        #         cbar.ax.tick_params(labelsize=16)
        #         # plot_contours(ax)
        #         # ax.plot([.35], [.5], 'x', color='red', ms=10, mew=3, ls='none')
        #         # ax.plot(np.tile(alpha_save, len(beta_save)),
        #         #         np.repeat(beta_save, len(alpha_save)),
        #         #         'rx')
        #         pt.show()
        #         # save_plot(fig, ax, f"BathymetricGradient_{labels[ind]}_CanyonWidth={canyon_width:.1f}km_{norm}",
        #         #           my_loc=2,
        #         #           folder_name="Parameter Sweeps/Figures/Parameters/solution")
    
        ### Vertical Velocity ###
        labels = ['Total'] # 'Along-shore', 'Cross-shore', 
        v_mins = [0.02, 0.02, 0]
        v_maxs = [0.165, 0.4, 6.5]
        for key in ['solution']: #'change of magnitude', 'magnitude of change', 
            for ind, key2 in enumerate(['w']): #'w_x', 'w_y', 
                for i, norm in enumerate(['L1', 'L2', 'Linf']):
                    fig, ax = plot_setup('$\\alpha$', '$\\beta$', scale=.7)#,
                                          # title=f"{labels[ind]} Vertical Speed (cm/s)")
                    A, B = np.meshgrid(alpha_values, beta_values)
                    vals = np.array([dictionaries[(alpha, beta)][key][key2][norm] \
                                      for alpha, beta in \
                                          zip(A[1:].flatten(), B[1:].flatten())]).reshape(A[1:].shape)
                    vals_slope = np.array([dictionaries['slope'][key][key2][norm] \
                                      for alpha in alpha_values])[None, :]
                    vals = np.concatenate([vals_slope, vals], axis=0)
                    c = ax.contourf(vals, origin='lower',
                                  extent=[alpha_values[0], alpha_values[-1],
                                          beta_values[0], beta_values[-1]],
                                  cmap='inferno_r',
                                  levels=np.linspace(v_mins[i], v_maxs[i], 21),
                                  extend='both'
                                  ) #vmin=v=_mins[i], vmax=v_maxs[i])#, vmin=1, vmax=8)
                    ax.set_aspect('equal')
                    cbar = add_colorbar(c,
                                        format=tick.FormatStrFormatter('%.2f'))
                    cbar.ax.tick_params(labelsize=16)
                    
                    cbar.ax.set_ylabel('Vertical velocity ($\\rm{cm/s}$)', rotation=270,
                            fontsize=16, labelpad=20)
                    # plot_contours(ax)
                    # ax.plot([.35], [.5], 'x', color='red', ms=10, mew=3, ls='none')
                    # ax.plot(np.tile(alpha_save, len(beta_save)),
                    #         np.repeat(beta_save, len(alpha_save)),
                    #         'rx')
                    # pt.show()
                    
                    save_plot(fig, ax, f"{labels[ind]}_VerticalVelocity_CanyonWidth={canyon_width:.1f}km_{norm}",
                              my_loc=2,
                              folder_name=f"Parameter Sweeps/Figures/Parameters/{key.replace(' ', '_')}")

if __name__ == "__main__":
    from barotropicSWEs.Configuration import configure

    param = configure.main()
    param.order = 3
    param.bbox_dimensional = (-100, 100, 0, 200)
    param.bbox = tuple([x * 1e3 / param.L_R for x in param.bbox_dimensional])
    alpha_values = np.round(np.linspace(.02, 0.98, 49), 3) #10, 91, 901
    beta_values = np.round(np.linspace(0, 1, 51), 3)
    param.canyon_depth = 2100
    
    for param.canyon_width in [5, 10, 15, 20]:
        parameter_run(
                param,
                param.canyon_width,
                alpha_values,
                beta_values,
                order=param.order,
                numerical_flux="Central",
                goal='NORMS' #Either plot solutions, plot perturbations or plot norms
        )
