#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sun Aug 22 12:27:53 2021

"""
import numpy as np


def startup(
    bbox,
    param,
    h_min,
    h_max,
    order,
    mesh=None,
    mesh_name="",
    canyon_width=1e-3,
    canyon_depth=1.0,
    canyon_length=3e-2,
    canyon_choice="v-shape",
    plot_domain=False,
    boundary_conditions="SOLID WALL",
    scheme="Lax-Friedrichs",
    potential_forcing=False,
    background_flow="Kelvin",
    max_tidal_amplitude=1,
    plot_topography=False,
    θ=0.5,
    rotation=True,
    show_exact_kelvin=False,
    wave_frequency=1.4,
    wavenumber=1.4,
    coastal_shelf_width=0.02,
    coastal_lengthscale=0.03,
    rayleigh_friction=0.05,
    verbose=True,
):

    import pickle
    from DGFEM.dgfem import FEM

    background_flow = background_flow.upper()
    assert boundary_conditions.upper() in [
        "SOLID WALL",
        "OPEN FLOW",
        "MOVING WALL",
        "SPECIFIED",
    ], "Invalid choice of boundary conditions."

    assert background_flow in [
        "KELVIN",
        "CROSSSHORE",
    ], "Invalid \
choice of background flow."

    assert scheme.upper() in [
        "CENTRAL",
        "UPWIND",
        "LAX-FRIEDRICHS",
        "RIEMANN",
        "PENALTY",
        "ALTERNATING",
    ], "Invalid choice of numerical flux."

    x0, xN, y0, yN = bbox
    Lx, Ly = xN - x0, yN - y0

    k, ω, r = wavenumber, wave_frequency, rayleigh_friction
    L0, λ = coastal_shelf_width, coastal_lengthscale
    ΔL, w, h_canyon = canyon_length, canyon_width, canyon_depth

    param.domain_size = (Lx, Ly)
    param.L_C = coastal_shelf_width * param.L_R
    param.L_S = (coastal_lengthscale - coastal_shelf_width) * param.L_R
    h_coastal = param.H_C / param.H_D

    from ppp.File_Management import file_exist, dir_assurer

    from barotropicSWEs.Configuration import topography

    h_func = topography.coastal_topography(
        slope_choice="smooth",
        canyon_choice=canyon_choice,
        shelf_depth=h_coastal,
        coastal_shelf_width=coastal_shelf_width,
        coastal_lengthscale=λ,
        canyon_length=ΔL,
        canyon_width=w,
        canyon_depth=h_canyon,
    )

    h_func_dim = lambda x, y: param.H_D * h_func(x, y)
    h_min = (λ - L0) / 10 if w < 1e-5 else min(w / 4, (λ - L0) / 10)

    if mesh is None:
        from MeshGeneration import make_canyon_meshes

        P, T, mesh_name = make_canyon_meshes.mesh_generator(
            bbox,
            param,
            h_min,
            h_max,
            mesh_dir="Meshes",
            canyon_func=h_func,
            canyon_width=w,
            canyon_length=ΔL,
            canyon_depth=h_canyon,
            plot_mesh=False,
            coastal_shelf_width=L0,
            coastal_lengthscale=λ,
            verbose=True,
        )

    else:
        P, T, mesh_name = mesh

    dir_assurer("Barotropic FEM Objects")
    fem_dir = f"Barotropic FEM Objects/{background_flow}_{mesh_name}_N={order}.pkl"

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

    if plot_domain:
        if verbose:
            print("Plotting domain")

        fig, ax = fem.PlotDomain2D()
        X, Y = np.linspace(x0, xN, 101), np.linspace(y0, yN, 101)
        X, Y = np.meshgrid(X, Y)
        H = h_func_dim(X, Y)

        cm = ax.matshow(H, origin="lower", aspect="auto", extent=bbox, alpha=0.2)
        cb = fig.colorbar(cm)
        cb.ax.set_title("Topography", y=0.4, x=3, rotation=270, fontsize=14)
        ax.set_title("")
        oFx = (fem.Fx.T.reshape(fem.Nfaces * fem.K, fem.Nfp)).T
        oFy = (fem.Fy.T.reshape(fem.Nfaces * fem.K, fem.Nfp)).T

        ax.plot(oFx, oFy, "k-", linewidth=0.2)
        ax.plot(
            fem.x.flatten("F"), fem.y.flatten("F"), "ro", markersize=2, label="Nodes"
        )
        eps = 0.05 * (xN - x0)
        ax.set_xlim(x0 - eps, xN + eps)
        ax.set_ylim(y0 - eps, yN + eps)
        ax.xaxis.set_ticks_position("bottom")
        ax.legend(fontsize=16, loc=4)

        import matplotlib.pyplot as pt

        pt.show()

    X, Y = np.round(fem.x.T.flatten(), 16), np.round(fem.y.T.flatten(), 16)
    if potential_forcing == False:
        φ = np.zeros(X.shape)

    else:
        φ = 0.34 * param.g * np.exp(1j * (0.7 * Y + k * X)) / (param.c ** 2)

    if background_flow == "KELVIN":
        kelvin_filename = f"ContinuousSlope_LC={param.L_C/param.L_R:.2e}_LS\
={param.L_S/param.L_R:.2e}_λ={λ:.2e}_delta={param.H_C/param.H_D:.2e}"
        x0 = bbox[0]
        slope_func = lambda y: h_func(x0, y)
        if verbose:
            print("Loading exact Perturbed Kelvin flow")

        u_kelv, v_kelv, η_kelv, ω, k = kelvin_solver(
            slope_func,
            param,
            coastal_lengthscale=λ,
            tidal_amplitude=max_tidal_amplitude,
            forcing_frequency=ω,
            foldername="Kelvin Flow",
            filename=f"{kelvin_filename}",
        )

        flow_bg = (
            u_kelv(X, np.round(Y, 10), 0),
            v_kelv(X, np.round(Y, 10), 0),
        )

    else:
        u_bg = lambda x, y: np.zeros(x.shape)
        v_bg = lambda x, y: 20 / (param.H_D * param.c) * np.ones(y.shape)
        flow_bg = (u_bg(X, Y), v_bg(X, Y))

    from barotropicSWEs import Barotropic

    swes = Barotropic.solver(
        fem,
        param,
        flux_scheme=scheme,
        θ=θ,
        boundary_conditions=boundary_conditions,
        rotation=True,
        background_flow=flow_bg,
        h_func=h_func,
        wave_frequency=ω,
        rayleigh_friction=r,
    )

    if plot_topography:
        if verbose:
            print("Plotting topography")

        from ppp.Plots import plot_setup
        import matplotlib.pyplot as pt

        fig, ax = plot_setup("x", "y")
        X, Y = np.linspace(x0, xN, 1001), np.linspace(y0, yN, 1001)
        X, Y = np.meshgrid(X, Y)
        H = h_func(X, Y)
        c = ax.matshow(
            -H,
            cmap="Blues_r",
            vmin=-1,
            vmax=-np.min(H),
            extent=[x0, xN, y0, yN],
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(c, ax=ax)
        pt.show()

        fig, ax = plot_setup("x", "y")
        X, Y = np.linspace(x0, xN, 1001), np.linspace(y0, yN, 1001)
        X, Y = np.meshgrid(X, Y)
        H = h_func(X, Y)
        c = ax.matshow(
            -H,
            cmap="Blues_r",
            vmin=-1,
            vmax=0,
            extent=[x0, xN, y0, yN],
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(c, ax=ax)

        # Plot mesh
        X, Y = P.T
        # ax.plot(X, Y, 'ro', markersize=1)
        ax.triplot(X, Y, T, "g-", linewidth=2, label="edges")

        ax.plot(
            fem.x.flatten("F"), fem.y.flatten("F"), "rx", markersize=10, label="Nodes"
        )

        pt.show()

    file_dir = f"{background_flow}_Frequency={ω:.1f}_scheme={scheme}_order={order}\
_CanyonWidth={w*param.L_R*1e-3:.2f}km_CanyonLength={ΔL*param.L_R*1e-3:.2f}km\
_CanyonDepth={h_canyon*param.H_D:.0f}m_CoastalLengthscale={λ*param.L_R*1e-3:.0f}km_\
PotentialForcing={potential_forcing}"

    return swes, φ, file_dir


def boundary_value_problem(
    swes, φ, animate=True, frames=20, file_name="", file_dir="BVP", verbose=True,
    save_solution=True,
):
    from ppp.Numpy_Data import save_arrays, load_arrays
    from ppp.File_Management import file_exist, dir_assurer
    import os

    

    file_name = (
        f"Flux={swes.scheme}_order={swes.fem.N}" if file_name == "" else file_name
    )
    Lx, Ly = swes.param.domain_size
    if np.round(Lx, 5) == np.round(Ly, 5):
        file_name += f"_DomainSize={Lx*swes.param.L_R*1e-3:.0f}km"

    else:
        file_name += f"_{Lx*swes.param.L_R*1e-3:.0f}kmx{Ly*swes.param.L_R*1e-3:.0f}km"

    if not file_exist(os.path.join(file_dir, file_name + ".npz")):
        print("File does not exist, generating data")
        sols = swes.boundary_value_problem(
            φ,
            animate=animate,
            frames=frames,
            file_name=os.path.join("BVP", "Animations", file_name),
            verbose=verbose,
        )
        sols = sols[0]
        if save_solution:
            dir_assurer(file_dir)
            save_arrays(file_name, (sols,), folder_name=file_dir)

    else:
        print("File exists, loading data")
        sols = load_arrays(file_name, folder_name=file_dir)[0]

    return sols


def kelvin_solver(
    h_func,
    param,
    coastal_lengthscale=0.05,
    tidal_amplitude=1,
    forcing_frequency=None,
    forcing_alongshore_wavenumber=None,
    foldername="Kelvin Flow",
    filename="",
    save=False,
):  # non-dimensional forcing frequency):
    import pickle
    from ppp.File_Management import file_exist, dir_assurer

    dir_assurer(foldername)
    k_, ω_ = forcing_alongshore_wavenumber, forcing_frequency
    λ_ = coastal_lengthscale
    ηF = tidal_amplitude

    if ω_ is not None:
        filename = f"{filename}_ω={ω_:.1f}.pkl"
        unknown = "Wavenumber"

    elif k_ is not None:
        filename = f"{filename}_k={k_:.1f}.pkl"
        unknown = "WaveFrequency"

    else:
        raise ValueError(
            "You must prescribe either a Kelvin wavenumber or \
wave frequency"
        )

    if not file_exist(f"{foldername}/{filename}"):
        from barotropicSWEs.Kelvin import kelvin_asymptotics

        if unknown == "WaveFrequency":
            kelvin = kelvin_asymptotics.Kelvin_Asymptotics(
                param,
                lambda y: param.H_D * h_func(y),
                ηF=ηF / param.H_D,
                λ=λ_,
                k=k_,
                ω=None,
            )

            kelvin.exact(k=k_)

        else:
            kelvin = kelvin_asymptotics.Kelvin_Asymptotics(
                param,
                lambda y: param.H_D * h_func(y),
                ηF=ηF / param.H_D,
                λ=λ_,
                k=None,
                ω=ω_,
            )

            kelvin.exact(ω=ω_)

        u, v, η = kelvin.sols_true
        y, ω, k = (
            kelvin.y,
            kelvin.ω_exact,
            kelvin.k_exact,
        )  # forcing frequency to be that of ω_PKW
        
        if save:
            with open(f"{foldername}/{filename}", "wb") as f:
                pickle.dump(
                    {
                        "u": u,
                        "v": v,
                        "η": η,
                        "y": y,
                        "ω": kelvin.ω_exact,
                        "k": kelvin.k_exact,
                    },
                    f,
                )

    else:
        with open(f"{foldername}/{filename}", "rb") as f:
            kelvin_dict = pickle.load(f)

        u, v, η, y = (
            kelvin_dict["u"],
            kelvin_dict["v"],
            kelvin_dict["η"],
            kelvin_dict["y"],
        )
        ω, k = (
            kelvin_dict["ω"],
            kelvin_dict["k"],
        )  # forcing frequency to be that of ω_PKW

    from scipy.interpolate import interp1d

    u_func, v_func = interp1d(y, u), interp1d(y, v)
    η_func = interp1d(y, η)

    u_kelv = lambda x, y, t: u_func(y) * np.exp(1j * (k * x - ω * t))
    v_kelv = lambda x, y, t: v_func(y) * np.exp(1j * (k * x - ω * t))
    η_kelv = lambda x, y, t: η_func(y) * np.exp(1j * (k * x - ω * t))

    return u_kelv, v_kelv, η_kelv, ω, k


def main(
    bbox,
    param,
    order=3,
    h_min=1e-3,
    h_max=5e-3,
    wave_frequency=1.4,
    wavenumber=1.4,
    coastal_lengthscale=0.03,
    frames=20,
    canyon_widths=[1e-10, 1e-3, 5e-3, 1e-2],
):
    ω, k, λ = wave_frequency, wavenumber, coastal_lengthscale
    Nt = frames

    for background_flow_ in ["KELVIN", "CROSSSHORE"]:  # , False]:
        for w_ in canyon_widths:
            for forcing_, r in zip([False], [0]):
                for scheme_ in ["Lax-Friedrichs"]:
                    swes, φ, file_dir = startup(
                        bbox,
                        param,
                        h_min,
                        h_max,
                        order,
                        mesh_name="",
                        canyon_width=w_,
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

                    boundary_value_problem(
                        swes,
                        φ,
                        frames=Nt,
                        file_dir="BVP",
                        file_name=file_dir,
                        animate=False,
                    )


if __name__ == "__main__":
    from barotropicSWEs.Configuration import configure

    param = configure.main()

    k = param.k * param.L_R  # non-dimensional alongshore wavenumber
    ω = param.ω / param.f  # non-dimensional forcing frequency

    # Non-dimensional canyon widths over which to iterate
    w_vals = np.linspace(1e-3, 1e-2, 19)  # [2::4]
    w_vals = np.insert(w_vals, 0, 0)

    param.bbox = (-param.domain / 2, param.domain / 2, 0, 0.05)
    main(
        param.bbox,
        param,
        param.order,
        param.hmin,
        param.hmax,
        coastal_lengthscale=(param.L_C + param.L_S) / param.L_R,
        canyon_widths=w_vals,
        wave_frequency=ω,
        wavenumber=k,
    )
