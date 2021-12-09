#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sun Aug 22 12:27:53 2021

"""
import numpy as np

def mesh_generator(
    bbox,
    param,
    h_min,
    h_max,
    canyon_func=None,
    mesh_name="",
    mesh_dir="Meshes",
    plot_mesh=True,
    canyon_width=5e-3,
    coastal_shelf_width=0.02,
    coastal_lengthscale=0.03,
    canyon_intrusion=.015,
    verbose=True,
):
    """
    Generates a 2D mesh of a box defined by the boundary box, bbox. The mesh
    considers the bathymetric gradient, ensuring a certain minimum number of 
    nodes. However, the user is recommended to provide both a minimum and a
    maximum edge size, h_min and h_max respectively.
    

    Parameters
    ----------
    bbox : tuple
        Boundary box of domain, given in Rossby radii.
    param : param class
        A class who attributes define the physical configuration.
    h_min : float
        Minimum edge size of discretised mesh.
    h_max : float
        Maximum edge size of discretised mesh.
    mesh_name : str, optional
        Filename of mesh. The default is "".
    canyon_width : float, optional
        Non-dimensional average width of submarine canyon. The default is 1e-3.
    coastal_shelf_width : TYPE, optional
        Non-dimensional width of the coastal shelf in the absence of a submarine
        canyon, given in Rossby radii. The default is 0.02.
coastal_lengthscale : float, optional
        Non-dimensional width of the coastal shelf and the continental margin
        in the absence of a submarine margin, given in Rossby radii. The default is 0.03.
    verbose : bool, optional
        Print progress. The default is True.

    Returns
    -------
    The points and the connectivity matrix of the discretised mesh.

    """

    x0, xN, y0, yN = bbox
    Lx, Ly = xN - x0, yN - y0
    L0, λ = coastal_shelf_width, coastal_lengthscale
    ΔL, w = canyon_intrusion, canyon_width

    param.domain_size = (Lx, Ly)
    param.L_C = coastal_shelf_width * param.L_R
    param.L_S = (coastal_lengthscale - coastal_shelf_width) * param.L_R

    if mesh_name == "":
        mesh_name = (
            f"DomainSize={Lx*param.L_R*1e-3:.0f}x{Ly*param.L_R*1e-3:.0f}km_\
h={h_max:.2e}_CanyonWidth={w*param.L_R*1e-3:.1f}km"
        )

    from ppp.File_Management import dir_assurer

    dir_assurer(mesh_dir)
    if canyon_func:
        h_func = lambda x, y: canyon_func(x, y)
        
    else:
        from topography import canyon_func1
        h_func = lambda x, y: canyon_func1(x, y, canyon_width=w,
                              coastal_shelf_width=L0, coastal_lengthscale=λ,
                              canyon_intrusion=ΔL)

    h_func_dim = lambda x, y: param.H_D * h_func(x, y)
    h_min = (λ - L0)/10 if w < 1e-5 else min(w/4, (λ - L0)/10)

    import uniform_box_mesh
    P, T = uniform_box_mesh.main(
        bbox,
        h_min,
        h_max,
        h_func=h_func_dim,
        edgefuncs=["Sloping"],
        folder=mesh_dir,
        file_name=mesh_name,
        plot_mesh=plot_mesh,
        verbose=True,
        plot_sdf=False,
        plot_boundary=False,
        save_mesh=True,
        max_iter=500,
        plot_edgefunc=False,
        slp=6,
        fl=0,
        wl=100,
    )

    return P, T, mesh_name

def main(bbox, param, h_min, h_max, coastal_lengthscale=.03,
         coastal_shelf_width=.02, canyon_intrusion=.015,
     canyon_widths=[0, 1e-3, 5e-3, 1e-2], mesh_name="", h_func=None):
    λ, LC = coastal_lengthscale, coastal_shelf_width
    ΔL = canyon_intrusion
    name_ = mesh_name
    for w in canyon_widths:
        h_func2 = lambda x, y: h_func(x, y, canyon_width=w, canyon_intrusion=ΔL,
                              coastal_shelf_width=LC, coastal_lengthscale=λ)
        mesh_generator(
            bbox,
            param,
            h_min,
            h_max,
            canyon_func=h_func2,
            mesh_name=name_,
            canyon_width=w,
            coastal_lengthscale=λ,
            coastal_shelf_width=LC,
            canyon_intrusion=ΔL,
            verbose=True,
        )

if __name__ == "__main__":
    h_min, h_max = 5e-4, 5e-2
    λ = 0.03 #Coastal Lengthscale

    from ChannelWaves1D.config_param import configure

    param = configure()
    param.H_D = 4000
    param.H_C = 200
    param.c = np.sqrt(param.g * param.H_D)
    param.f, param.ω = 1e-4, 1.4e-4
    param.L_R = param.c/abs(param.f)
    param.Ly = 2 * param.L_R
    k = param.k * param.L_R  # non-dimensional alongshore wavenumber
    ω = param.ω / param.f  # non-dimensional forcing frequency
    w_vals = np.linspace(1e-3, 1e-2, 19)
    w_vals = np.insert(w_vals, 0, 0)
    from topography import canyon_func1

    for domain_size in [.05, .1, .15]:
        bbox = (-domain_size/2, domain_size/2, 0, .05)
        main(bbox, param, h_min, h_max, coastal_lengthscale=λ,
             canyon_intrusion=.015, coastal_shelf_width=.02,
             canyon_widths=w_vals[:5], h_func=canyon_func1)