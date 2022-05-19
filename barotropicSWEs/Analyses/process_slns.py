#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sun Aug 22 12:27:53 2021

"""
import numpy as np


def plot_solutions(
    param,
    swes,
    sols,  # Dimensional Solutions
    wave_frequency=1.4e-4,  # in s^{-1}
    file_name="",
    file_dir="BVP",
    verbose=True,
    bbox_plot=(-100, 100, 0, 200),  # in kilometres
    zoom=True,
):
    """
    Plot and save the spatial structure of the simulation results.

    Parameters
    ----------
    swes : Barotropic
        A Barotropic class which holds our variable values.
    sols : numpy.array
        Spatial data corrsponding to Barotropic model.
    wave_frequency : float, optional
        Forcing wave-frequency given non-dimensionally with respect to the
        Coriolis cofficient. The default is 1.4, correpsonding to semi-diurnal
        frequency at mid-latitudes whereby f = 10^{-4} rad/s.
    file_name : str, optional
        Principal name of file output. The default is "".
    file_dir : str, optional
        Principal name of directory in which to save file output. The default
        is "BVP".
    verbose : bool, optional
        Print log of actions. The default is True.
    bbox_plot : float, optional
        The boundary box of domain if zoom is True.
    zoom : bool, optional
        Whether to plot and save spatial structure of the close-up, zoomed-in
        domain (True), or to plot the simulation domain (Fals). The default is
        True.

    Returns
    -------
    None.

    """

    from ppp.File_Management import dir_assurer, file_exist

    dir_assurer(file_dir)
    x0, xN, y0, yN = param.bbox_dimensional
    file_name = (
        f"Flux={swes.scheme}_order={swes.fem.N}" if file_name == "" else file_name
    )

    plot_dir = f"{file_dir}/{file_name}"

    if bbox_plot == param.bbox_dimensional:
        zoom = False

    vals, grid = gridded_data(
        swes, sols, tuple([1e3 * x / param.L_R for x in bbox_plot])
    )
    xg, yg = grid  # non-dimensional units in metres
    X, Y = np.meshgrid(xg * 1e-3, yg * 1e-3)

    # import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as pt

    if not zoom:
        from matplotlib import patches
    from ppp.Plots import plot_setup, save_plot

    dir_assurer(plot_dir)
    labels = [
        "u",
        "v",
        "p",
        "Qx",
        "Qy",
        "vorticity",
        "w",
        "uhx",
        "vhy",
        "KEx",
        "KEy",
        "KE",
        "PE",
    ]

    # if file_exist(f'{plot_dir}/u_0_zoom={zoom}.png'):
    #     return

    for i in range(len(vals)):
        val, plot_name = vals[i], labels[i]
        max_val = np.nanmax(np.abs(val))
        phases = [0] if plot_name in ["KEx", "KEy", "KE", "PE"] else [0, 1 / 4]
        for phase in phases:
            fig, ax = plot_setup("Along-shore ($\\rm{km}$)", "Cross-shore ($\\rm{km}$)")
            to_plot = np.real(np.exp(2j * np.pi * phase) * val)
            c = ax.matshow(
                to_plot,
                aspect="auto",
                cmap="seismic",
                extent=param.bbox_dimensional,
                origin="lower",
                vmin=-max_val,
                vmax=max_val,
            )
            
            ax.contour(
                X,
                Y,
                param.H_D * swes.h_func(X * 1e3 / param.L_R, Y * 1e3 / param.L_R),
                sorted([param.H_C + 1e-3, param.canyon_depth, param.H_D - 1e-3]),
                colors="black",
            )
            
            if not zoom:
                rect = patches.Rectangle(
                    (bbox_plot[0], bbox_plot[2]),
                    bbox_plot[1] - bbox_plot[0],
                    bbox_plot[3] - bbox_plot[2],
                    linewidth=3,
                    edgecolor="black",
                    facecolor="none",
                )
                ax.add_patch(rect)

            cbar = fig.colorbar(c, ax=ax)
            cbar.ax.tick_params(labelsize=16)
            ax.set_aspect("equal")
            save_plot(fig, ax, f"{plot_name}_{phase}_zoom={zoom}", folder_name=plot_dir)
            pt.close("all")

    u = 1e-2 * vals[0][::40, ::40]
    v = 1e-2 * vals[1][::40, ::40]
    p = vals[2]
    max_val = np.nanmax(np.abs(p))
    import matplotlib.pyplot as pt

    for phase in np.arange(8) / 8:
        fig_vel, ax_vel = plot_setup(
            x_label="Along-shore (km)", y_label="Cross-shore (km)"
        )
        fig_sol, ax_sol = plot_setup(
            x_label="Along-shore (km)", y_label="Cross-shore (km)"
        )

        c = ax_sol.contourf(
            (p * np.exp(2j * np.pi * phase)).real,
            cmap="seismic",
            alpha=0.5,
            extent=param.bbox_dimensional,
            origin="lower",
            levels=np.linspace(-max_val, max_val, 21),
        )
        cbar = fig_sol.colorbar(c, ax=ax_sol)
        cbar.ax.tick_params(labelsize=16)

        for fig, ax, name, x_pos in zip(
            [fig_vel, fig_sol], [ax_vel, ax_sol], ["Velocity", "Solution"], [0.7, 0.89]
        ):
            ax.contour(
                X,
                Y,
                param.H_D * swes.h_func(X * 1e3 / param.L_R, Y * 1e3 / param.L_R),
                sorted([param.H_C + 1e-3, param.canyon_depth, param.H_D - 1e-3]),
                colors="black",
            )
            Q = ax.quiver(
                X[::40, ::40],
                Y[::40, ::40],
                (u * np.exp(2j * np.pi * phase)).real,
                (v * np.exp(2j * np.pi * phase)).real,
                width=0.002,
                scale=1,
            )

            qk = ax.quiverkey(
                Q,
                x_pos,
                0.05,
                0.05,
                r"$5\,\rm{cm/s}$",
                labelpos="W",
                coordinates="figure",
                fontproperties={"weight": "bold", "size": 18},
            )
            ax.set_aspect("equal")
            save_plot(fig, ax, f"{name}_{phase}_zoom={zoom}", folder_name=plot_dir)

    if verbose:
        print(f"Finished plotting in {plot_dir}")


def gridded_data(swes, sols, bbox_plot, Nx=1001, Ny=1001):
    """
    Generates quantities constructed from Barotropic model solutions, and
    intrpolates the quantities onto a rgularly-spaced grid of size Nx x Ny.

    Parameters
    ----------
    swes : Barotropic
        A Barotropic class which holds our variable values.
    sols : numpy.array
        Spatial data corrsponding to Barotropic model.
    bbox_plot : TYPE
        DESCRIPTION.
    Nx : int, optional
        Number of regularly-space nodes to discretise along-shore
        domain. The default is 1001.
    Ny : int, optional
        Number of regularly-space nodes to discretise cross-shore
        domain. The default is 1001.

    Returns
    -------
    tuple
        A tuple of model variables, along with the regularly-spaced grids on
        which these variables have been interpolated.

    """
    x, y = np.round(swes.X, 15), np.round(swes.Y, 15)
    u, v, p = np.split(sols, 3, axis=0)

    from scipy.interpolate import griddata

    gridded_vals = []
    xg, yg = np.linspace(bbox_plot[0], bbox_plot[1], Nx), np.linspace(
        bbox_plot[2], bbox_plot[3], Ny
    )
    X, Y = np.meshgrid(xg, yg)
    dx = swes.param.L_R * (bbox_plot[1] - bbox_plot[0]) / Nx
    dy = swes.param.L_R * (bbox_plot[3] - bbox_plot[2]) / Ny
    for val in [u, v, p, swes.h]:
        gridded_vals.append(griddata((x, y), val, (X, Y), method="cubic"))

    from barotropicSWEs.Configuration.topography import grad_function

    ρ = swes.param.ρ_ref
    c, g = swes.param.c, swes.param.g
    (
        u,
        v,
        p,
        h,
    ) = gridded_vals  # Gridded solution non-dimensional system variables, and fluid depth
    u, v, p, h = (
        100 * u * c,
        100 * v * c,
        1e-3 * ρ * (c ** 2) * p,
        swes.param.H_D * h,
    )  # Dimensionalised quantities
    H = swes.param.H_D * swes.h_func(
        X, Y
    )  # Dimensional fluid depth projected on mesh grid
    ux, uy = grad_function(u*1e-2, dx, dy)  # spatial derivatives of along-shore velocity
    vx, vy = grad_function(v*1e-2, dx, dy)  # spatial derivatives of cross-shore velocity
    hx, hy = grad_function(H, dx, dy)  # bathymetric gradients
    Qx, Qy = h * u*1e-2, h * v*1e-2  # Along-shore and cross-shore volume fluxes, respectivelu
    vorticity = vx - uy  # Vorticity
    u_gradh = -(u * hx + v * hy)  # w = - u . grad(h)
    KEx = 0.5 * ρ * h * u * u.conjugate() * 1e-4  # Along-shore kinetic energy
    KEy = 0.5 * ρ * h * v * v.conjugate() * 1e-4  # Cross-shore kinetic energy
    PE = 0.5 * p * p.conjugate()/(ρ * g)  # Potential energy
    KE = KEx + KEy  # Total kinetic energy

    vals = (u, v, p, Qx, Qy, vorticity, u_gradh, u * hx, v * hy, KEx, KEy, KE, PE)

    return vals, (xg * swes.param.L_R, yg * swes.param.L_R)


def main(
    bbox,
    param,
    order=3,
    h_min=1e-3,
    h_max=5e-3,
    wave_frequency=1.4,
    wavenumber=1.4,
    rayleigh_friction=0,
    coastal_lengthscale=0.03,
    canyon_widths=[0, 1e-3, 5e-3, 1e-2],
    canyon_intrusion=0.015,
    potential_forcing=False,
    scheme="Lax-Friedrichs",
):
    """
    Generates simulation data and plots (as well as saves) spatial structure of
    simulation results.

    Parameters
    ----------
    bbox : tuple
        Boundary box of domain.
    param : parameter
        A dictionary of default parameter values for system set-up.
    order : int, optional
        Polynomial order of local test function in DG-FEM numerics. The
        default is 3.
    h_min : float, optional
        Minimum size of mesh element for edge-size function for mesh
        generation. The default is 1e-3.
    h_max : float, optional
        Maximum size of mesh element for edge-size function for mesh
        generation. The default is 5e-3.
    wave_frequency : float, optional
        Forcing wave-frequency given non-dimensionally with respect to the
        Coriolis cofficient. The default is 1.4, correpsonding to semi-diurnal
        frequency at mid-latitudes whereby f = 10^{-4} rad/s.
    wavenumber : float, optional
        Forcing wavenumber given non-dimensionally with respect to the
        Rossby radius of deformation. The default is 1.4, correpsonding to
        the modal wavenumber of the trivial Kelvin wave forced at semi-diurnal
        frequency.
    rayleigh_friction : float, optional
        Linear Rayligh friction coefficient given non-dimensionally with
        respect to the Coriolis cofficient. The default is 0.
    coastal_lengthscale : float, optional
        This is th total non-dimensional lengthscale of the coastal topography
        consisting of the shelf and the continental slope. The default is
        0.03, corresponding to around 60 km, which according to Harris and
        Whiteway, 2013, is th average contiental-shelf lengthscale.
    canyon_widths : iterable, optional
        An iterable of non-dimensional canyon widths in terms of Rossby radii.
        The default is [0, 1e-3, 5e-3, 1e-2], which corresponds to
        [0, 2, 10, 20] km.
    potential_forcing : bool, optional
        The choice of whether to prescribe additional external potential
        forcing. The default is False.
    scheme : str, optional
        Numerical flux scheme in DG-FEM numerics. The default is
        'Lax-Friedrichs', which is essntially the continuous form of the exact
        Riemann flux.

    """

    from SWEs import startup, boundary_value_problem

    ω, k, r = wave_frequency, wavenumber, rayleigh_friction
    λ, ΔL = coastal_lengthscale, canyon_intrusion
    forcing_, scheme_ = potential_forcing, scheme
    for background_flow_ in ["KELVIN", "CROSSSHORE"]:  # ,
        for w_ in canyon_widths:
            print(order, background_flow_, w_)
            swes, φ, filename = startup(
                bbox,
                param,
                h_min,
                h_max,
                order,
                mesh_name="",
                canyon_width=w_,
                canyon_intrusion=ΔL,
                plot_domain=False,
                boundary_conditions="Specified",
                scheme=scheme_,
                potential_forcing=forcing_,
                background_flow=background_flow_,
                plot_topography=False,
                θ=0.5,
                rotation=True,
                show_exact_kelvin=False,
                wave_frequency=ω,
                wavenumber=k,
                coastal_lengthscale=λ,
                rayleigh_friction=r,
            )

            sols = boundary_value_problem(
                swes,
                φ,
                file_dir="BVP",
                file_name=filename,
                animate=False,
            )

            for zoom_ in [False, True]:
                plot_solutions(
                    swes,
                    sols,
                    wave_frequency=ω,
                    file_name=filename,
                    file_dir="BVP",
                    verbose=True,
                    L_zoom=0.05,
                    zoom=zoom_,
                )


if __name__ == "__main__":
    h_min, h_max = 5e-4, 5e-2
    λ = 0.03

    from ChannelWaves1D.config_param import configure

    param = configure()
    param.H_D = 4000
    param.H_C = 200
    param.c = np.sqrt(param.g * param.H_D)
    param.f, param.ω = 1e-4, 1.4e-4
    param.L_R = param.c / abs(param.f)
    param.Ly = 2 * param.L_R
    k = param.k * param.L_R  # non-dimensional alongshore wavenumber
    ω = param.ω / param.f  # non-dimensional forcing frequency
    w_vals = np.linspace(1e-3, 1e-2, 19)[3::5]
    w_vals = np.insert(w_vals, 0, 0)

    for domain_width in [0.1]:
        bbox = (-domain_width / 2, domain_width / 2, 0, 0.05)
        for order in [3]:
            main(
                bbox,
                param,
                order,
                h_min,
                h_max,
                coastal_lengthscale=λ,
                canyon_widths=w_vals[::-1],
                wave_frequency=ω,
                wavenumber=k,
            )
