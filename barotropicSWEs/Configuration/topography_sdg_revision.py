#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Tue Apr 26 08:58:36 2022

"""
import matplotlib.pyplot as pt, numpy as np


def phi(X):
    """
    Produces non-dimensional slope profile.

    Parameters
    ----------
    X : float/numpy.ndarray
        Non-dimensional (w.r.t slope length) cross-shore coordinates.

    Returns
    -------
    float/numpy.ndarray
        Non-dimensional slope profile.

    """

    return np.sin(X * np.pi / 2) ** 2


def hss(y, pars):
    """
    Produces the slope/shelf profile in the absence of a canyon.

    Parameters
    ----------
    y : float/numpy.ndarray
        Cross-shore coordinates.
    pars : parameter class
        Contains slope/canyon parameters

    Returns
    -------
    float/numpy.ndarray
        Slope/shelf profile.

    """

    # depth of shelf/slope/abyss
    h1, h2 = pars.h1, pars.h2
    L_C, L_S = pars.L_C, pars.L_S

    return (
        h1 * (y < L_C)
        + (y >= L_C) * (y <= L_C + L_S) * (h1 + (h2 - h1) * phi((y - L_C) / L_S))
        + h2 * (y > L_C + L_S)
    )


def hc(y, x, pars):
    """
    Produces canyon profile.

    Parameters
    ----------
    y : float/numpy.ndarray
        Cross-shore coordinates.
    x : float/numpy.ndarray
        Along-shore coordinates.
    pars : parameter class
        Contains slope/canyon parameters

    Returns
    -------
    float/numpy.ndarray
        Canyon profile along continental margin.

    """

    h1, h2 = pars.h1, pars.h2
    W, L = pars.W, pars.L
    L_S = pars.L_S
    L_CS, L_CC = pars.L_CS, pars.L_CC
    beta = pars.beta

    # d: cross-canyon lengthscale used in formula!
    d = (W/2) * np.sqrt(1 + h1 * (1 + L_CS / L_CC) / ((h2 - h1) * phi(beta)))

    return hv(y, pars) * (1 - (x / d) ** 2)


def hv(y, pars):
    """
    Produces a linear valley profile at the centre of the canyon.

    Parameters
    ----------
    x : float/numpy.ndarray
        Cross-shore coordinates (in km's)
    pars : parameter class
        Contains slope/canyon parameters

    Returns
    -------
    float/numpy.ndarray
        valley profile at centre of canyon.

    """
    # depth of valley (linear in x), at centre of canyon

    h1, h2 = pars.h1, pars.h2
    L, L_CC, L_CS = pars.L, pars.L_CC, pars.L_CS
    L_C = pars.L_C
    beta = pars.beta

    return (
        (y >= L_C - L_CC)
        * (y <= L_C + L_CS)
        * (h1 + (h2 - h1) * phi(beta) * (y - L_C * (1 - pars.alpha)) / L)
    )

def coastal_topography(param):
    class pars:
        L_S = param.L_S  # slope width (km)
        L_C = param.L_C  # shelf width (km)
        W = param.canyon_width * 1e3  # (half) canyon width AT SHELF BREAK (km)
        alpha = param.alpha  # 0<alpha<1: canyon occupies this proportion of shelf (ND)
        beta = param.beta  # 0<beta<1: canyon occupies this proportion of slope (ND)
        L_CC, L_CS = alpha * L_C, beta * L_S
        L = L_CC + L_CS
        h1 = param.H_C  # shelf depth (m)
        h2 = param.H_D  # open-ocean depth (m)
        
    h_slope = lambda y : hss(y, pars)
    
    if param.alpha < 1e-2 or param.beta < 1e-2 or param.canyon_width < 1:
        h_canyon = lambda x, y : h_slope(y)
    else:
        h_canyon = lambda x, y : hc(y, x, pars)
    
    topography = lambda x, y : (y > pars.L_C + pars.L_S) * h_slope(y) + \
        (y <= pars.L_C + pars.L_S) * (
        (h_slope(y) < h_canyon(x, y)) * h_canyon(x, y) + (h_slope(y) >= h_canyon(x, y)) * h_slope(y)
    )
        
    return h_slope, topography

if __name__ == "__main__":
    class pars:
        L_S = 50  # slope width (km)
        L_C = 100  # shelf width (km)
        W = 5  # (half) canyon width AT SHELF BREAK (km)
        alpha = 0.5  # 0<alpha<1: canyon occupies this proportion of shelf (ND)
        beta = 0.2  # 0<beta<1: canyon occupies this proportion of slope (ND)
        L_CC, L_CS = alpha * L_C, beta * L_S
        L = L_CC + L_CS
        h1 = 200  # shelf depth (m)
        h2 = 4000  # open-ocean depth (m)
        
    

    nx, ny = 500, 500
    y = np.linspace(0, 200, ny + 1)  # Cross-shore coordinates (km)

    # pt.figure(1, figsize=[5, 4])
    # pt.xlabel("Cross-shore (km)")
    # pt.ylabel("Bathymetry (m)")
    from ppp.Plots import plot_setup, save_plot
    fig, ax = plot_setup("Cross-shore (km)",
                         "Bathymetry (m)", scale=.65)
    ax.plot(y, -hss(y, pars), "k", label='Slope')
    ax.plot(y, -hv(y, pars), "r", label='Valley')
    ax.set_ylim([-1.1 * pars.h2, 50])
    pt.show()
    # save_plot(fig, ax, f'CrossSection_alpha={pars.alpha}_beta={pars.beta}',
    #           folder_name="Topographic Profiles", my_loc=1)

    x = np.linspace(-15, 15, nx + 1)  # Along-shore coordinates (km)
    xa, ya = np.meshgrid(x, y)
    hssa = hss(ya, pars)
    hca = hc(ya, xa, pars)

    # take maximum of hss (shelf/slope geometry) and hc (canyon geometry),
    # provided we are on the shelf or slope:
    h = (ya > pars.L_C + pars.L_S) * hssa + (ya <= pars.L_C + pars.L_S) * (
        (hssa < hca) * hca + (hssa >= hca) * hssa
    )

    fig = pt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Along-shore (km)")
    ax.set_zticks(np.linspace(-4000, 0, 5))
    ax.set_ylabel("Cross-shore (km)")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Bathymetry (m)", rotation=90)
    ax.plot_surface(xa, ya, -h, cmap="YlGnBu_r")
    ax.plot_wireframe(xa, ya, -h, rstride=10, cstride=10,
                      color="black", linewidth=0.25)
    ax.view_init(20, 65)
    fig.tight_layout()
    pt.show()
    # save_plot(fig, ax, f'Canyon_alpha={pars.alpha}_beta={pars.beta}',
    #           folder_name="Topographic Profiles")