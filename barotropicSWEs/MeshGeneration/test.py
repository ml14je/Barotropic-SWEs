#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Sun Oct  3 01:30:40 2021
"""

import numpy as np


def create_shoreline_geometry(x0, xN, y0, yN, name="poly"):
    import geopandas as gpd
    import pandas as pd

    box = _bbox(x0, y0, xN, yN)

    gpd.GeoDataFrame(
        pd.DataFrame(["p1"], columns=["geom"]),
        crs={"init": "epsg:4326"},
        geometry=[box],
    ).to_file(name)


def _bbox(x0, y0, xN, yN):
    from shapely.geometry import Polygon

    arr = np.array([[x0, y0], [x0, yN], [xN, yN], [xN, y0]])
    return Polygon(arr)


def main(
    bbox,
    h_min,
    h_max,
    edgefuncs="Uniform",
    folder="",
    file_name="",
    plot_mesh=False,
    h_func=None,
    verbose=False,
    plot_sdf=True,
    plot_boundary=True,
    save_mesh=True,
    max_iter=50,
    canyon_kwargs=None,
    plot_edgefunc=False,
    HD=4000,
    wl=100,
    slp=10,
    fl=0,
    mesh_gradation=.2
):
    file_name = "Mesh" if file_name == "" else file_name

    x0, xN, y0, yN = bbox
    LR = np.sqrt(9.81 * HD) * 1e4 #Rossby radius of deformation - horizontal lengthscale

    if not file_exist(f"{folder}/{file_name}.npz"):
        from oceanmesh import (
            Shoreline,
            fix_mesh,
            distance_sizing_function,
            generate_mesh,
            laplacian2,
            make_mesh_boundaries_traversable,
            feature_sizing_function,
            create_bbox,
            enforce_mesh_gradation,
            wavelength_sizing_function,
            compute_minimum,
            Region,
            bathymetric_gradient_sizing_function,
            DEM,
        )
        EPSG = 4326  # EPSG code for WGS84 which is what you want to mesh in
        # Specify and extent to read in and a minimum mesh size in the unit of the projection

        extent = Region(extent=bbox, crs=EPSG)
        if type(edgefuncs) == str:
            edgefuncs = [edgefuncs.upper()]

        edgefuncs = [edgefunc.upper() for edgefunc in edgefuncs]
        x0, xN, y0, yN = bbox
        shp_folder = "Box_Shp"
        dir_assurer(f"{folder}/{shp_folder}")
        fname = f"{folder}/{shp_folder}/box_shape_({x0},{y0})x({xN},{yN}).shp"
        eps = h_max
        create_shoreline_geometry(x0, xN, y0 - eps, y0, name=fname)
        extent_ = Region((x0, xN, y0 - eps / 2, yN), crs=EPSG)
        points_fixed = np.array([[x0, y0], [x0, yN], [xN, yN], [xN, y0]])
        
        if h_func is None: #By default, it chooses the canyon topography
            from topography import canyon_func1

            if canyon_kwargs is not None and type(canyon_kwargs) == dict:
                #h_min = min(canyon_kwargs['w']/20, h_min) #You force to hav at least 10 grid points over the width of canyon
                h_func = lambda x, y: HD * canyon_func1(x, y, **canyon_kwargs)

            else:
                h_func = lambda x, y: HD * canyon_func1(x, y)
        print("Barotropic h_min", h_min, extent.bbox)
        dem = DEM(h_func, bbox=extent.bbox, resolution=h_min)  # creates a DEM object for own function, h_func
        
        smoothing = False
        print("creating shoreline")
        if len(edgefuncs) == 1 and edgefuncs[0] == "UNIFORM":
            shore = Shoreline(fname, extent_.bbox, h_max,
                              smooth_shoreline=smoothing)

        else:
            shore = Shoreline(fname, extent_.bbox, h_min,
                              smooth_shoreline=smoothing)

        if verbose:
            print("created shoreline")

        domain = create_bbox(extent.bbox)  # signed_distance_function(shore)

        if verbose:
            print("created a signed distance function")

        edge_lengths = []

        if "UNIFORM" in edgefuncs:
            edge_lengths.append(
                distance_sizing_function(shore, max_edge_length=h_max, rate=0)
            )

        if "DISTANCING" in edgefuncs:
            edge_lengths.append(distance_sizing_function(shore,
                                                         max_edge_length=h_max))

        if "FEATURES" in edgefuncs:
            edge_lengths.append(
                feature_sizing_function(shore, domain, max_edge_length=h_max)
            )

        if "WAVELENGTH" in edgefuncs:
            edge_lengths.append(wavelength_sizing_function(dem, wl=wl*LR))

        if "SLOPING" in edgefuncs:
            h_min *= 50
            edge_lengths.append(bathymetric_gradient_sizing_function(
                dem, slope_parameter=slp,
                filter_quotient=fl,
                min_edge_length=h_min,
                max_edge_length=h_max))

        if len(edge_lengths) > 1:
            edge_length = compute_minimum(edge_lengths)

        else:
            edge_length = edge_lengths[0]

        # Enforce gradation - becomes very unnatural otherwise!
        # Becomes distorted if dx != dy in grid
        edge_length = enforce_mesh_gradation(edge_length,
                                              gradation=mesh_gradation)

        points, cells = generate_mesh(
            domain, edge_length,
            max_iter=max_iter,
            pfix=points_fixed,
            lock_boundary=True,
        )
        # mesh_plot(points, cells, '1 Original Mesh', folder, f_points=points_fixed)

        # Makes sure the vertices of each triangle are arranged in a counter-clockwise order
        points, cells, jx = fix_mesh(points, cells)

        # remove degenerate mesh faces and other common problems in the mesh
        points, cells = make_mesh_boundaries_traversable(points, cells)
        
        # Make sure nodal points lie on bbox
        X, Y = points.T
        X[abs(X - xN) < h_min*1e-3] = xN
        X[abs(X - x0) < h_min*1e-3] = x0
        Y[abs(Y - y0) < h_min*1e-3] = y0
        Y[abs(Y - yN) < h_min*1e-3] = yN
        points = np.array([X, Y]).T

        if edgefuncs != "UNIFORM":
            points, cells = laplacian2(points, cells)  # Final poost-processed mesh

        if save_mesh:
            save_arrays(file_name, [points, cells], folder_name=folder)

    else:
        points, cells = load_arrays(file_name, folder_name=folder)

    if file_exist(f"{folder}/{file_name}.png"):
        save_mesh = False
        
    else:
        save_mesh = True

    if plot_mesh:
        for zoom_ in [True, False]:
            mesh_plot(points, cells, file_name, folder=folder, f_points=None,
                      h_func=h_func, save=save_mesh, zoom=zoom_)

    return points, cells


def mesh_plot(points, cells, name, folder="", f_points=None, h_func=None,
              save=True, L_R=2000, zoom=True, aspect="auto", linewidth=.2):
    fig, ax = plot_setup("Along-shore (km)", "Cross-shore (km)")
    X, Y = L_R*points.T

    if h_func:
        x0, y0, xN, yN = np.min(X), np.min(Y), np.max(X), np.max(Y)
        x, y = np.linspace(x0, xN, 1001), np.linspace(y0, yN, 1001)
        xg, yg = np.meshgrid(x, y)
        c = ax.matshow(
            h_func(xg/L_R, yg/L_R),
            cmap="Blues",
            vmin=0,
            extent=[x0, xN, y0, yN],
            aspect=aspect,
            origin="lower",
            alpha=.7
        )
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.tick_params(labelsize=16)
        
    if f_points is not None:
        X, Y = L_R*f_points.T
        ax.plot(X, Y, "go", markersize=2)
    

    ax.plot(X, Y, "gx", markersize=1)
    ax.triplot(X, Y, cells, linewidth=linewidth, color="red")
    ax.set_aspect('equal')
    
    if zoom:
        ax.set_xlim([-100, 100])
        ax.set_ylim([0, 250])
        name += '_zoom'

    if save:
        save_plot(fig, ax, name, folder_name=folder)
        
    else:
        import matplotlib.pyplot as pt
        pt.show()

def dir_assurer(folder_name, wd=''):
    import os

    if not os.path.exists(
        os.path.join(
            wd, folder_name
            )
        ):
        os.makedirs(
            os.path.join(
                wd, folder_name
                )
            )
            
def file_exist(file_name, folder_name='', wd=''):
    import os

    return os.path.isfile(
        os.path.join(
            wd, os.path.join(
                folder_name, file_name
                )
            )
        )

def save_arrays(file_name, arrays, wd=None,
                folder_name=None):
    import os

    for array in arrays:
        assert array.dtype ==  'float64' or 'complex128'

    if not wd: wd = os.getcwd()
    if not folder_name: folder_name = ''

    folder_name = os.path.join(
        wd, folder_name
        )
    dir_assurer(folder_name)

    np.savez_compressed(os.path.join(
            folder_name, file_name+'.npz'
            ),
                        *arrays
                        )

def load_arrays(file_name, wd=None, folder_name=None):
    import os
    if not wd: wd = os.getcwd()
    if not folder_name: folder_name = ''

    loaded = np.load(
        os.path.join(
            wd, os.path.join(
                folder_name, file_name+'.npz'
                )
            ), allow_pickle=True
        )

    return tuple([loaded[key] for key in loaded])

def plot_setup(x_label='', y_label='', x_log=False, y_log=False, bx=10, by=10,
               y_rev=False, scale=1, title='', my_dpi=100, project=None,
               colour_scheme='seaborn-colorblind'):
    """
    Sets up figure environment.

    Parameters
    ----------
    x_label : String
        Label of the x axis.
    y_label : String
        Label of the y axis.
    x_log : Boolean
        Determines whether the x axis is log scale.
    y_log : Boolean
        Determines whether the y axis is log scale.
    bx : Float
        The base of the x axis if x_log == True.
    by : Float
        The base of the y axis if y_log == True.
    scale : Float
        Scale of figure environment with respect to an aspect ratio of 13.68
        x 7.68.
    title : String
        Title of figure plot.
    dpi : Integer
        Density of pixels per square inch of figure plot.
    """
    import matplotlib.pyplot as pt
    import matplotlib as mpl
    
    
    mpl.style.use(colour_scheme)
    fig, ax = pt.subplots(figsize=(13.68*scale, 7.68*scale),
                          dpi=my_dpi)
    set_axis(ax, x_label, y_label, x_log, y_log, bx, by,
               y_rev, scale, title)

    return fig, ax

def set_axis(ax, x_label='', y_label='', x_log=False, y_log=False, bx=10, by=10,
               y_rev=False, scale=1, title=''):
    import matplotlib.pyplot as pt
    pt.gca()
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_title(title, fontsize=22)
    if x_log: ax.set_xscale('log', base=bx)
    if y_log: ax.set_yscale('log', base=by)
    # print(y_rev)
    # if y_rev: ax.invert_yaxis()
    ax.grid(linewidth=0.5)

def save_plot(fig, ax, file_name, folder_name=None, wd=None, my_dpi=100,
              my_loc='best', give_legend=True, labels=None,
              override_check=False):
    import os
    import matplotlib.pyplot as pt

    if not wd: wd = os.getcwd()

    def replace_all(text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    def save(handles=None):
        if give_legend:
            try:
                handles, labels2 = ax.get_legend_handles_labels()

            except:
                AttributeError

            if labels:
                ax.legend(labels, fontsize=16, loc=my_loc)

            elif handles: ax.legend(fontsize=16, loc=my_loc)

        fig.savefig(os.path.join(
                wd, os.path.join(
                        folder_name, file_name
                        )
                ),
                bbox_inches='tight', dpi=my_dpi)
        pt.close(fig)

    if folder_name:
        dir_assurer(folder_name, wd)
    else:
        folder_name = ''

    # file_name = replace_all(
    #         file_name, {'\\' : '', '$' : '', ' ' : '_', ',' : '', '.' : ','}
    #         )

    file_name += '.png'
    
    if not override_check:
        save()
        return

    if not file_exist(file_name, folder_name, wd):
        save()

    else:
        print('File already exists.')
        answer = input('\nWould you like to override old plot? Y for yes, and \
N for no: ').upper()

        if answer == 'Y':
            save()

        elif answer == 'N':
            return

        else:
            print('\nInvalid entry. Plot has not been saved.')
            return

if __name__ == "__main__":
    h_min, h_max = 5e-3, 5e-2

    class param:
        g = 9.81
        H_D = 4000
        H_C = 200
        L_C = 100e3
        L_S = 50e3
        c = np.sqrt(g * H_D)
        f, ω = 1e-4, 1.4e-4
        L_R = c/abs(f)
        Ly = 2 * L_R
        k = 7.7e-7

    k = param.k * param.L_R  # non-dimensional alongshore wavenumber
    ω = param.ω / param.f  # non-dimensional forcing frequency
    w_vals = np.linspace(1e-3, 1e-2, 19)
    w_vals = np.insert(w_vals, 0, 0)
    
    
    def phi(Y):
        return np.cos(np.pi * Y / 2)**2
    
    def topography(x, y):
        
        return (param.H_C * (y * param.L_R <= param.L_C) + \
                (param.L_R * y > param.L_C) * \
                    (param.L_R * y < param.L_C + param.L_S) * \
                    (param.H_D - (param.H_D - param.H_C) * \
                        phi((param.L_R * y - param.L_C)/param.L_S)) + \
                        param.H_D * (y * param.L_R >= param.L_C + param.L_S))/param.H_D
            
    h_func_dim = lambda x, y : param.H_D * topography(x, y)


    P, T = main(
        (-5e-2, 5e-2, 0, 1e-1),
        h_min,
        h_max,
        h_func=h_func_dim,
        edgefuncs=["Sloping"],
        folder="Meshes",
        file_name="trial",
        plot_mesh=False,
        verbose=True,
        plot_sdf=False,
        plot_boundary=False,
        save_mesh=True,
        max_iter=500,
        plot_edgefunc=False,
        slp=28,
        fl=0,
        wl=100,
        mesh_gradation=.2
    )
    
    mesh_plot(
        P,
        T,
        "trial",
        folder="Meshes/Figures",
        h_func = lambda x, y : param.H_D * topography(x, y),
        save=True,
        L_R=param.L_R * 1e-3,
        zoom=False,
    )