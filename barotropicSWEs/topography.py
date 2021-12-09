#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Mon Nov  30 15:33:46 2021

"""
import numpy as np

@np.vectorize
def uniform_func(x, y):
    """
    Gives an array of ones, signifying that the fluid depth is constant.

    Parameters
    ----------
    x : np.array
        Along-shore horizontal variable on which to evaluate the fluid depth
        function.
    y : np.array
        Cross-shore horizontal variable on which to evaluate the fluid depth
        function.

    Returns
    -------
    H : np.array or float
        Fluid depth of 1, regardless of position.

    """
    
    return 1.0


@np.vectorize
def canyon_func(x, y, shelf_depth=0.05, coastal_shelf_width=.02,
                coastal_lengthscale=.03, canyon_intrusion=.015,
                canyon_width=5e-3, α=4):
    """
    This is the function we use to model a continuous and smooth submarine
    canyon along a continental margin, along an individual to control a range
    of parameters which define the feature's geometry, such as the width,
    slope, protrusion and the steepness of the canyon sidewalls.

    Parameters
    ----------
    x : np.array
        Along-shore horizontal variable on which to evaluate the fluid depth
        function.

    y : np.array
        Cross-shore horizontal variable on which to evaluate the fluid depth
        function.

    shelf_depth : float, optional
        Non-dimensional coastal fluid depth w.r.t the deep-ocean fluid depth.
        The default is .025.

    coastal_shelf_width : float, optional
        Shelf-width in the absence of a submarine canyon, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. The default is .02.
        
    coastal_lengthscale : float, optional
        Coastal width length scale in the absence of a submarine canyon, which
        is the sum of the slope width and the shelf width. This is given
        non-dimensionally w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation. The default is .03.

    canyon_intrusion : float, optional
        Depth of canyon protrusion, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. Note that a positive value indicates an intrusion of
        the coastal shelf, while a negative value indicates an extrusion. The
        default is .015.

    canyon_width : float, optional
        Averag canyon width w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation.. The default is 5e-3.

    α : float, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    H : np.array or float
        Fluid depth evaluated at x, y

    """
    L0, λ = coastal_shelf_width, coastal_lengthscale
    ΔL, w = canyon_intrusion, canyon_width
    h = shelf_depth
    
    ΔH = 1 - h  # ΔH is change in fluid depth between shelf and deep-ocean
    # Gradient change of Gaussian shelf-width profile
    # Slope-width parameter (Teeluck, 2013)

    LS = λ - L0  # slope width in absence of canyon

    x0, xN = np.min(x), np.max(x)
    xm = 0.5 * (x0 + xN)
    xC = xm  # Center of canyon in x

    if w < 1e-5:  # .02 # Width of canyon for α >> 1
        LC = L0  # no canyon
    else:
        LC = L0 - ΔL * np.exp(
            -((2 * (x - xC) / w) ** α) / 2
        )  # Shelf-width variance in x

    if y <= LC:
        H = h

    elif (y > LC) and (y < LS + LC):
        H = 1.0 - ΔH * np.cos(np.pi * (y - LC) / (2 * LS)) ** 2

    else:
        H = 1.0

    return H


@np.vectorize
def canyon_func1(x, y, shelf_depth=0.05, coastal_shelf_width=.02,
                coastal_lengthscale=.03, canyon_intrusion=.015,
                canyon_width=5e-3, canyon_centre=0):
    """
    This is the function we use to model a continuous sloping submarine
    canyon along a continental margin, allowing the individual to control a
    range of parameters which define the feature's geometry, such as the width,
    slope, protrusion and the steepness of the canyon sidewalls.

    Parameters
    ----------
    x : np.array or float
        Along-shore horizontal variable on which to evaluate the fluid depth
        function.

    y : np.array or float
        Cross-shore horizontal variable on which to evaluate the fluid depth
        function.

    shelf_depth : float, optional
        Non-dimensional coastal fluid depth w.r.t the deep-ocean fluid depth.
        The default is .025.

    coastal_shelf_width : float, optional
        Shelf-width in the absence of a submarine canyon, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. The default is .02.
        
    coastal_lengthscale : float, optional
        Coastal width length scale in the absence of a submarine canyon, which
        is the sum of the slope width and the shelf width. This is given
        non-dimensionally w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation. The default is .03.

    canyon_intrusion : float, optional
        Depth of canyon intrusion, given non-dimensionally w.r.t the dominant
        horizontal lengthscale L_R that is the Rossby radius of deformation.
        The default is .015.

    canyon_width : float, optional
        Maximum canyon width w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation. The default is 5e-3.
        
    canyon_centre : float, optional
        Centre along-shore position of canyon given non-dimensionally w.r.t
        the Rossby radius of deformation, L_R. The default is 0.

    Returns
    -------
    H : np.array or float
        Fluid depth evaluated at x, y

    """
    L0, λ = coastal_shelf_width, coastal_lengthscale
    LS = λ - L0 #Slope width away from canyon
    ΔL, w = canyon_intrusion, canyon_width
    h = shelf_depth
    xm = canyon_centre
    
    ΔH = 1.0 - h  # ΔH is change in fluid depth between shelf and deep-ocean
    if w > 1e-5:
        if abs(x-xm) <= w/2:
            LS_val = LS + ΔL - 2 * abs(x - xm) * ΔL / w
            LC_val = L0 - ΔL + 2 * abs(x - xm) * ΔL / w
            
        else:
            LS_val = LS
            LC_val = L0
            
    else:
        LS_val = LS
        LC_val = L0

    if y <= LC_val:
        H = h

    elif (y > LC_val) and (y < λ):
        H = 1.0 - ΔH * np.cos(np.pi * (y - LC_val) / (2 * LS_val))**2

    else:
        H = 1.0

    return H


@np.vectorize
def canyon_func2(x, y, shelf_depth=0.05, coastal_shelf_width=.02,
                coastal_lengthscale=.03, canyon_intrusion=.015,
                canyon_width=5e-3):
    """
    This is the function we use to model a discontinuous flat-bottom submarine
    canyon along a continental margin, along an individual to control a range
    of parameters which define the feature's geometry, such as the width,
    slope, protrusion and the steepness of the canyon sidewalls.

    Parameters
    ----------
    x : np.array
        Along-shore horizontal variable on which to evaluate the fluid depth
        function.

    y : np.array
        Cross-shore horizontal variable on which to evaluate the fluid depth
        function.

    shelf_depth : float, optional
        Non-dimensional coastal fluid depth w.r.t the deep-ocean fluid depth.
        The default is .025.

    coastal_shelf_width : float, optional
        Shelf-width in the absence of a submarine canyon, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. The default is .02.
        
    coastal_lengthscale : float, optional
        Coastal width length scale in the absence of a submarine canyon, which
        is the sum of the slope width and the shelf width. This is given
        non-dimensionally w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation. The default is .03.

    canyon_intrusion : float, optional
        Depth of canyon protrusion, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. Note that a positive value indicates an intrusion of
        the coastal shelf, while a negative value indicates an extrusion. The
        default is .015.

    canyon_width : float, optional
        Averag canyon width w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation.. The default is 5e-3.

    Returns
    -------
    H : np.array or float
        Fluid depth evaluated at x, y

    """
    L0, λ = coastal_shelf_width, coastal_lengthscale
    ΔL, w = canyon_intrusion, canyon_width
    h = shelf_depth
    
    ΔH = 1.0 - h  # ΔH is change in fluid depth between shelf and deep-ocean

    LS = λ - L0  # slope width in absence of canyon        

    if y <= L0:
        H = h

    elif (y > L0) and (y < LS + L0):
        H = 1.0 - ΔH * np.cos(np.pi * (y - L0) / (2 * LS)) ** 2

    else:
        H = 1.0
        
    if ((λ > y) & (y > λ-ΔL))  & ((-w*(y-λ+ΔL)/(2*ΔL) < x) & (x < w*(y-λ+ΔL)/(2*ΔL))):
        H = 1.0

    return H


@np.vectorize
def canyon_func3(x, y, shelf_depth=0.05, coastal_shelf_width=.02,
                coastal_lengthscale=.03, canyon_intrusion=.015,
                canyon_width=5e-3):
    """
    This is the function we use to model a continuous near-critical submarine
    canyon along a continental margin, along an individual to control a range
    of parameters which define the feature's geometry, such as the width,
    slope, protrusion and the steepness of the canyon sidewalls.

    Parameters
    ----------
    x : np.array
        Along-shore horizontal variable on which to evaluate the fluid depth
        function.

    y : np.array
        Cross-shore horizontal variable on which to evaluate the fluid depth
        function.

    shelf_depth : float, optional
        Non-dimensional coastal fluid depth w.r.t the deep-ocean fluid depth.
        The default is .025.

    coastal_shelf_width : float, optional
        Shelf-width in the absence of a submarine canyon, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. The default is .02.
        
    coastal_lengthscale : float, optional
        Coastal width length scale in the absence of a submarine canyon, which
        is the sum of the slope width and the shelf width. This is given
        non-dimensionally w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation. The default is .03.

    canyon_intrusion : float, optional
        Depth of canyon protrusion, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. Note that a positive value indicates an intrusion of
        the coastal shelf, while a negative value indicates an extrusion. The
        default is .015.

    canyon_width : float, optional
        Averag canyon width w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation.. The default is 5e-3.

    Returns
    -------
    H : np.array or float
        Fluid depth evaluated at x, y

    """
    L0, λ = coastal_shelf_width, coastal_lengthscale
    ΔL, w = canyon_intrusion, canyon_width
    h = shelf_depth
    
    ΔH = 1.0 - h  # ΔH is change in fluid depth between shelf and deep-ocean
    # Gradient change of Gaussian shelf-width profile
    # Slope-width parameter (Teeluck, 2013)

    LS = λ - L0  # slope width in absence of canyon

    if y <= L0:
        H = h

    elif (y > L0) and (y <= LS + L0):
        H = 1.0- ΔH * np.cos(np.pi * (y - L0) / (2 * LS)) ** 2

    else:
        H = 1.0
        
    p0, p1 = np.array([0, λ, 1]), np.array([0, λ-ΔL, h])
    p2, p3 = np.array([-w/2, L0, h]), np.array([w/2, L0, h])
    
    
    
    N1, N2 = p2 - p0, p2 - p1
    n = np.array([N1[1] * N2[2] - N1[2] * N2[1],
                  N1[2] * N2[0] - N1[0] * N2[1],
                  N1[0] * N2[1] - N1[1] * N2[0]])

    if (((λ >= y) & (y >= L0))  & ((-w/2 * (1 -(y-L0)/(λ-L0)) <= x) & (x <= 0)) | \
                    ((L0 >= y) & (y >= λ-ΔL))  & ((-w/2 * (0 + (y-λ+ΔL)/(L0-ΔL)) <= x) & (x <= 0))):
        H = -((n[0] * (x - p0[0]) + n[1] * (y - p0[1]) - n[2] * p0[2])/n[2])
        
    N1, N2 = p3 - p0, p3 - p1
    n = np.array([N1[1] * N2[2] - N1[2] * N2[1],
                  N1[2] * N2[0] - N1[0] * N2[1],
                  N1[0] * N2[1] - N1[1] * N2[0]])  
    
    if (((λ >= y) & (y >= L0))  & ((w/2 * (1 -(y-L0)/(λ-L0)) >= x) & (x >= 0)) | \
                    ((L0 >= y) & (y >= λ-ΔL))  & ((w/2 * (0 +(y-λ+ΔL)/(L0-ΔL)) >= x) & (x >= 0))):
        H = -((n[0] * (x - p0[0]) + n[1] * (y - p0[1]) - n[2] * p0[2])/n[2])

    return H

def grad_function(H, dy, dx):
    """
    earth_gradient(F,HX,HY), where F is 2-D, uses the spacing
    specified by HX and HY. HX and HY can either be scalars to specify
    the spacing between coordinates or vectors to specify the
    coordinates of the points.  If HX and HY are vectors, their length
    must match the corresponding dimension of F.
    """
    hy, hx = np.zeros(H.shape, dtype=H.dtype), np.zeros(H.shape, dtype=H.dtype)

    # Forward diferences on edges
    hx[:, 0] = (H[:, 1] - H[:, 0]) / dx
    hx[:, -1] = (H[:, -1] - H[:, -2]) / dx
    hy[0, :] = (H[1, :] - H[0, :]) / dy
    hy[-1, :] = (H[-1, :] - H[-2, :]) / dy

    # Central Differences on interior
    hx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / (2 * dx)
    hy[1:-1, :] = (H[2:, :] - H[:-2, :]) / (2 * dy)

    return hx, hy

def plot_topography(bbox, param, func, coastal_lengthscale=.03,
                    coastal_shelf_width=.02, canyon_intrusion=.01,
                    canyon_width=5e-3, plot_name='', plot_type='CONTOUR'):
    x0, xN, y0, yN = bbox
    LC, λ = coastal_shelf_width, coastal_lengthscale
    ΔL, w = canyon_intrusion, canyon_width
    x, y = np.linspace(x0, xN, 101), np.linspace(y0, yN, 101)
    X, Y = np.meshgrid(x, y)

    H = param.H_D * func(X, Y, shelf_depth=param.H_C/param.H_D,
                          coastal_shelf_width=LC,
                          coastal_lengthscale=λ,
                          canyon_intrusion=ΔL,
                          canyon_width=w)
    
    from ppp.Plots import plot_setup

    if plot_type.upper()=='CONTOUR':
        fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
        levels = np.linspace(200.001, 3999.999, 11)
        CS  = ax.contour(X*param.L_R*1e-3, Y*param.L_R*1e-3,
                         H, levels, cmap='jet_r')
        ax.clabel(CS, inline=1, fontsize=14)
        
    elif plot_type.upper() == 'MESH':
        import matplotlib.pyplot as pt
        from mpl_toolkits.mplot3d import axes3d
        fig = pt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Along-shore (km)')
        ax.set_zticks(np.linspace(-4000, 0, 5))
        ax.set_ylabel('Cross-shore (km)')
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('Elevation (m)', rotation=90,)# labelpad=10)
        ax.plot_wireframe(X*param.L_R*1e-3, Y*param.L_R*1e-3,
                          -H, rstride=30,
                  cstride=30, color='blue', linewidth=.5)
        ax.view_init(30, 40)
        fig.tight_layout()
    
    else:
        fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
        c = ax.matshow(
            H,
            cmap="Blues",
            vmax=param.H_D,
            vmin=0,
            extent=[x0*param.L_R*1e-3, xN*param.L_R*1e-3,
                    y0*param.L_R*1e-3, yN*param.L_R*1e-3],
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(c, ax=ax)

    if plot_name:
        from ppp.Plots import save_plot
        if plot_name.upper() == 'SAVE':
            plot_name = f'Canyon_{plot_type}'
        save_plot(fig, ax, plot_name)
        
    else:
        import matplotlib.pyplot as pt
        pt.show()

if __name__ == "__main__":
    h_min, h_max = 5e-4, 5e-2
    λ = 0.03

    from ChannelWaves1D.config_param import configure

    param = configure()
    param.H_D = 4000
    param.H_C = 200
    param.c = np.sqrt(param.g * param.H_D)
    param.f, param.ω = 1e-4, 1.4e-4
    param.L_R = param.c/abs(param.f)
    w = 0.009 # 5e-3

    for domain_width in [.1]:
        bbox = (-domain_width/2, domain_width/2, 0, domain_width)
        plot_topography(bbox, param, canyon_func1, coastal_lengthscale=λ,
                        coastal_shelf_width=.02, canyon_intrusion=.015,
                        canyon_width=w, plot_type='batheymtry', plot_name='')
