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
                coastal_lengthscale=.03, canyon_length=.015,
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

    canyon_length : float, optional
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
    ΔL, w = canyon_length, canyon_width
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
def canyon_func1(x, y,
                 shelf_depth=0.05,
                 coastal_shelf_width=.02,
                 coastal_lengthscale=.03,
                 canyon_width=5e-3,
                 canyon_length=3e-2,
                 canyon_depth=1.0, 
                 canyon_centre=0,
                 canyon_head_depth=0.05):
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

    canyon_length : float, optional
        Depth of canyon intrusion, given non-dimensionally w.r.t the dominant
        horizontal lengthscale L_R that is the Rossby radius of deformation.
        The default is .015.

    canyon_width : float, optional
        Maximum canyon width w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation. The default is 5e-3.
        
    canyon_centre : float, optional
        Centre along-shore position of canyon given non-dimensionally w.r.t
        the Rossby radius of deformation, L_R. The default is 0.
        
    canyon_depth : float, optional
        Depth within canyon at the foot of the slope. The default is None, in
        which case, it becomes the depth at the deep ocean (non-dimensionally,
        1.0).

    Returns
    -------
    H : np.array or float
        Fluid depth evaluated at x, y

    """
    L0, λ = coastal_shelf_width, coastal_lengthscale
    LS = λ - L0 #Slope width away from canyon
    ΔL, w = canyon_length, canyon_width
    h = shelf_depth
    xm = canyon_centre
    h2 = 1.0 if canyon_depth is None else canyon_depth
    h3 = canyon_head_depth
    
    
    ΔH = 1.0 - h  # ΔH is change in fluid depth between shelf and deep-ocean
    ΔH2 = h2 - h3  # ΔH is change in fluid depth between shelf and foot of canyon
    
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
        if abs(x) > w/2:
            H = 1.0 - ΔH * np.cos(np.pi * (y - LC_val) / (2 * LS_val))**2
            
        else:
            H = h2 - ΔH2 * np.cos(np.pi * (y - LC_val) / (2 * LS_val))**2

    else:
        H = 1.0

    return H


@np.vectorize
def v_canyon(x, y,
             slope_function,
             shelf_depth=0.05,
             coastal_shelf_width=.02,
             coastal_lengthscale=.03,
             canyon_length=.015,
             canyon_width=5e-3,
             canyon_foot=.03,
             canyon_foot_depth=1.0,
             canyon_head_depth=0.05):
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

    canyon_length : float, optional
        Depth of canyon protrusion, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. Note that a positive value indicates an intrusion of
        the coastal shelf, while a negative value indicates an extrusion. The
        default is .015.

    canyon_width : float, optional
        Averag canyon width w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation. The default is 5e-3.
        
    canyon_depth : float, optional
        Depth within canyon at the foot of the slope. The default is None, in
        which case, it becomes the depth at the deep ocean (non-dimensionally,
        1.0).

    Returns
    -------
    H : np.array or float
        Fluid depth evaluated at x, y

    """
    L0, λ = coastal_shelf_width, canyon_foot
    ΔL, w = canyon_length, canyon_width
    # print(2000*canyon_width)
    # L_R = 2000
    # print(L0*2000, 2000*λ, ΔL*2000, w*2000)
    # raise ValueError
    h = shelf_depth
    # L_canyon = canyon_foot if canyon_foot is not None else canyon_foot
    h3 = canyon_head_depth
    h4 = canyon_foot_depth
    
    # print(canyon_foot, canyon_depth)
    
    
    ΔH = 1.0 - h  # ΔH is change in fluid depth between shelf and deep-ocean
    ΔH2 = h4 - h3   # ΔH is change in fluid depth between head and foot of canyon

    LS = coastal_lengthscale - L0  # slope width in absence of canyon   
    
    if abs(x) < w/2:
        LC_val = L0 + (λ - ΔL - L0) * (1 - 2 * abs(x)/ w)
        LC_val = max(LC_val, λ - ΔL)
        LS_val = λ - LC_val

    else:
        LC_val = L0
        LS_val = LS
        

    if y <= L0:
        H = h

    elif (y > L0) and (y < LS + L0):
        H = slope_function(y)

    else:
        H = 1.0
        
    if (abs(x) < w/2):
        if max(LC_val, λ-ΔL) < y <= λ:
            H = h4 - ΔH2 * np.cos(np.pi * (y - LC_val) / (2 * LS_val)) ** 2

    return H


def find_slope_intersect(slope_function,
                         canyon_depth,
                         shelf_width,
                         coastal_lengthscale,
                         initial_value=None):
    from ppp.Newton_Raphson import newton_raphson
    
    slope_width = coastal_lengthscale - shelf_width
    
    initial_value = shelf_width + slope_width/2 if initial_value is None else \
        initial_value
    
    def f(y):
        return slope_function(y) - canyon_depth

    canyon_foot = newton_raphson(f, initial_value)
    assert (canyon_foot >= shelf_width) and (canyon_foot <= coastal_lengthscale)
    
    return canyon_foot

@np.vectorize
def diamond_canyon(x, y,
             slope_function,
             shelf_depth=0.05,
             coastal_shelf_width=.02,
             coastal_lengthscale=.03,
             canyon_length=.015,
             canyon_width=5e-3,
             canyon_depth=None,
             canyon_foot=None):
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

    canyon_length : float, optional
        Depth of canyon protrusion, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. Note that a positive value indicates an intrusion of
        the coastal shelf, while a negative value indicates an extrusion. The
        default is .015.

    canyon_width : float, optional
        Averag canyon width w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation. The default is 5e-3.

    canyon_depth : float, optional
        Depth within canyon at the foot of the slope. The default is None, in
        which case, it becomes the depth at the deep ocean (non-dimensionally,
        1.0).

    Returns
    -------
    H : np.array or float
        Fluid depth evaluated at x, y

    """
    
    L0, λ = coastal_shelf_width, coastal_lengthscale
    λ0 = canyon_foot if canyon_foot is not None else λ
    ΔL, w = canyon_length, canyon_width
    h = shelf_depth
    h2 = canyon_depth if canyon_depth is not None else 1.0

    LS = λ - L0  # slope width in absence of canyon
    if y <= L0:
        H = h

    elif (y > L0) and (y <= LS + L0):
        H = slope_function(y)

    else:
        H = 1.0

    if w > 0:
        p0, p1 = np.array([0, λ0, h2]), np.array([0, λ0-ΔL, h])
        p2, p3 = np.array([-w/2, L0, h]), np.array([w/2, L0, h])
        
        N1, N2 = p2 - p0, p2 - p1
        n = np.array([N1[1] * N2[2] - N1[2] * N2[1],
                      N1[2] * N2[0] - N1[0] * N2[1],
                      N1[0] * N2[1] - N1[1] * N2[0]])
    
        if (((λ0 >= y) & (y >= L0))  & ((-w/2 * (1 -(y-L0)/(λ0-L0)) < x) & (x <= 0)) | \
                        ((L0 >= y) & (y >= λ0-ΔL))  & ((-w/2 * (0 + (y-λ0+ΔL)/(L0-λ0+ΔL)) < x) & (x <= 0))):
            H = max(-((n[0] * (x - p0[0]) + n[1] * (y - p0[1]) - n[2] * p0[2])/n[2]), H)
            
        N1, N2 = p3 - p0, p3 - p1
        n = np.array([N1[1] * N2[2] - N1[2] * N2[1],
                      N1[2] * N2[0] - N1[0] * N2[1],
                      N1[0] * N2[1] - N1[1] * N2[0]])  
        
        if (((λ0 >= y) & (y >= L0))  & ((w/2 * (1 -(y-L0)/(λ0-L0)) > x) & (x >= 0)) | \
                        ((L0 >= y) & (y >= λ0-ΔL))  & ((w/2 * (0 +(y-λ0+ΔL)/(L0-λ0+ΔL)) > x) & (x >= 0))):
            H = max(-((n[0] * (x - p0[0]) + n[1] * (y - p0[1]) - n[2] * p0[2])/n[2]), H)

    return H

def smooth_slope(y,
                 shelf_depth=0.05,
                 coastal_shelf_width=.02,
                 coastal_lengthscale=.03
                 ):
    L0 = coastal_shelf_width
    LS = coastal_lengthscale - L0 # slope width
    h = shelf_depth
    
    ΔH = 1.0 - h  # ΔH is change in fluid depth between shelf and deep-ocean
    
    return 1.0 - ΔH * np.cos(np.pi * (y - L0) / (2 * LS)) ** 2

def linear_slope(y,
                 shelf_depth=0.05,
                 coastal_shelf_width=.02,
                 coastal_lengthscale=.03
                 ):
    L0 = coastal_shelf_width
    LS = coastal_lengthscale - L0 # slope width
    h = shelf_depth
    
    return h + (1 - h) * (y - L0)/LS

def GG07(y,
         shelf_depth=0.05,
         coastal_shelf_width=.02,
         coastal_lengthscale=.03,
         relative_density_difference=1e-2,
         fixed_upper_layer_depth=2.5e-2
         ):
    
    c_inf = np.sqrt(relative_density_difference * fixed_upper_layer_depth)
    cC = c_inf * np.sqrt(1 - fixed_upper_layer_depth/shelf_depth)
    cD = c_inf * np.sqrt(1 - fixed_upper_layer_depth)
    LC = coastal_shelf_width
    λval = coastal_lengthscale
    LS = λval - LC


    # save_plot(fig, ax, f'wf={ω}_s={s:.2e}_r={r:.2e}',
    #           folder_name='Asymptotic Energy Flux', my_loc=2)
    
    @np.vectorize
    def c(x):
        if x < LC:
            vals = cC
        elif x > λval:
            vals = cD
            
        else:
            vals = cC + (cD - cC) * (x - LC)/LS

        return vals

    return fixed_upper_layer_depth/(1 - (c(y)/c_inf)**2)
    
def coastal_topography(param,
                       slope_choice='smooth',
                       canyon_choice='v-shape',
                       shelf_depth=0.05,
                       coastal_shelf_width=.02,
                       coastal_lengthscale=.03,
                       canyon_length=.015,
                       canyon_width=5e-3,
                       canyon_foot_depth=1.0,
                       smooth=True,
                       compare_filtering=False):
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

    canyon_length : float, optional
        Depth of canyon protrusion, given non-dimensionally
        w.r.t the dominant horizontal lengthscale L_R that is the Rossby radius
        of deformation. Note that a positive value indicates an intrusion of
        the coastal shelf, while a negative value indicates an extrusion. The
        default is .015.

    canyon_width : float, optional
        Averag canyon width w.r.t the dominant horizontal lengthscale L_R that
        is the Rossby radius of deformation. The default is 5e-3.

    canyon_depth : float, optional
        Depth within canyon at the foot of the slope. The default is None, in
        which case, it becomes the depth at the deep ocean (non-dimensionally,
        1.0).

    Returns
    -------
    H : np.array or float
        Fluid depth evaluated at x, y

    """
    
    if slope_choice.upper() == 'COS_SQUARED':
        slope_function = lambda y : smooth_slope(y, 
                                                 shelf_depth,
                                                 coastal_shelf_width,
                                                 coastal_lengthscale
                                                 )
        
    elif slope_choice.upper() == 'LINEAR':
        slope_function = lambda y : linear_slope(y, 
                                                 shelf_depth,
                                                 coastal_shelf_width,
                                                 coastal_lengthscale
                                                 )
    elif slope_choice.upper() == 'GG07':
        slope_function = lambda y : GG07(
            y, 
            shelf_depth,
            coastal_shelf_width,
            coastal_lengthscale,
            relative_density_difference=param.reduced_gravity/param.g,
            fixed_upper_layer_depth=param.H_pyc/param.H_D
            )
    
    else:
        raise ValueError("Incorrect choice of slope function.")
        
    def slope_topography(y):
        return shelf_depth * (y < coastal_shelf_width) + \
            (y >= coastal_shelf_width) * (y <= coastal_lengthscale) * slope_function(y) + \
                (y > coastal_lengthscale)

    if round(canyon_width, 5) == 0 or round(canyon_foot_depth, 5) == 1:
        canyon_foot = coastal_lengthscale

    else:
        canyon_foot = find_slope_intersect(slope_function,
                                           canyon_foot_depth,
                                           coastal_shelf_width,
                                           coastal_lengthscale)

    canyon_head_depth = slope_topography(canyon_foot - canyon_length)
    print(canyon_foot * 1e-3 * param.L_R, (canyon_head_depth + canyon_foot) * 1e-3 * param.L_R/2)

    if canyon_width < 1e-4 or \
        canyon_length < 1e-4 or \
            abs(canyon_foot_depth - shelf_depth) < 1e-3:
        canyon_topography = lambda x, y : slope_topography(y)
        
    elif canyon_choice.upper() == 'DIAMOND-SHAPE':
        canyon_topography = lambda x, y : diamond_canyon(x, y,
                                            slope_function,
                                            shelf_depth,
                                            coastal_shelf_width,
                                            coastal_lengthscale,
                                            canyon_length,
                                            canyon_width,
                                            canyon_foot,
                                            canyon_foot_depth,
                                            canyon_head_depth)
        
    elif canyon_choice.upper() == 'V-SHAPE':
        canyon_topography = lambda x, y : v_canyon(x, y,
                                            slope_function,
                                            shelf_depth,
                                            coastal_shelf_width,
                                            coastal_lengthscale,
                                            canyon_length,
                                            canyon_width,
                                            canyon_foot,
                                            canyon_foot_depth,
                                            canyon_head_depth)
    else:
        canyon_topography = None
        
    if smooth:
        canyon_width = max(canyon_width, .5 * 1e3/param.L_R)
        Lx = max(1.5 * canyon_width, 20 * 1e3/param.L_R)
        x0, xN = -Lx/2, Lx/2
        y0, yN = 0, coastal_lengthscale + 20 * 1e3/param.L_R
        res = param.resolution
    
        x_fine, y_fine = np.arange(x0, xN+res, res), np.arange(y0, yN+res, res)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        H_fine = canyon_topography(X_fine, Y_fine)
    
        from ppp.filterfx import filt2
        H_filtered = filt2(H_fine, res, 20*res, "lowpass")
    
        if compare_filtering:
            from ppp.Plots import plot_setup 
            for H in [H_fine, H_filtered]:
                fig, ax = plot_setup('Along-shore (km)',
                                     'Cross-shore (km)')
                LR = 1e-3 * param.L_R
                import matplotlib.pyplot as pt
                c = ax.matshow(param.H_D*H,
                               aspect='auto',
                               cmap='Blues',
                               extent=[x0 * LR, xN * LR, y0 * LR, yN * LR],
                               origin='lower',
                               vmin=param.H_C, vmax=param.H_D)
                fig.colorbar(c, ax=ax)
                ax.set_ylim([100, 140])
                pt.show()
    
        from scipy.interpolate import RectBivariateSpline as interpolate
        canyon_topography_int = interpolate(y_fine, x_fine, H_filtered)
    
        def canyon_topography(x, y):
            return canyon_topography_int.ev(y, x)
        
        def slope_topography(y):
            return canyon_topography_int.ev(y, 100 * 1e3/param.L_R)
    
    return slope_topography, canyon_topography

def grad_function(H, dy, dx, derivative=1):
    """
    earth_gradient(F,HX,HY), where F is 2-D, uses the spacing
    specified by HX and HY. HX and HY can either be scalars to specify
    the spacing between coordinates or vectors to specify the
    coordinates of the points.  If HX and HY are vectors, their length
    must match the corresponding dimension of F.
    """
    hy, hx = np.zeros(H.shape, dtype=H.dtype), np.zeros(H.shape, dtype=H.dtype)

    if derivative == 1:
        # Forward diferences on edges
        hx[:, 0] = (H[:, 1] - H[:, 0]) / dx
        hx[:, -1] = (H[:, -1] - H[:, -2]) / dx
        hy[0, :] = (H[1, :] - H[0, :]) / dy
        hy[-1, :] = (H[-1, :] - H[-2, :]) / dy
    
        # Central Differences on interior
        hx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / (2 * dx)
        hy[1:-1, :] = (H[2:, :] - H[:-2, :]) / (2 * dy)
        
    elif derivative == 2:
        hx[:, 0] = (H[:, 0] -2 * H[:, 1] + H[:, 2]) / (dx * dx)
        hx[:, -1] = (H[:, -1] -2 * H[:, -2] + H[:, -3]) / (dx * dx)
        
        hy[0, :] = (H[0, :] -2 * H[1, :] + H[2, :]) / (dy * dy)
        hy[-1, :] = (H[-1, :] -2 * H[-2, :] + H[-3, :]) / (dy * dy)

        # Central Differences on interior
        hx[:, 1:-1] = (H[:, 2:] -2 * H[:, 1:-1] + H[:, :-2]) / (dx * dx)
        hy[1:-1, :] = (H[2:, :] -2 * H[1:-1, :] + H[:-2, :]) / (dy * dy)
        
    else:
        raise ValueError('Invalid order of derivative')
        

    return hx, hy

def plot_topography(bbox, param,
                    slope_choice='SMOOTH',
                    canyon_choice='V-SHAPE',
                    coastal_lengthscale=.03,
                    coastal_shelf_width=.02,
                    coastal_shelf_depth=.05,
                    canyon_length=.01,
                    canyon_width=5e-3,
                    canyon_foot_depth=.5,
                    plot_name='',
                    smooth=True,
                    plot_type='CONTOUR'):

    assert plot_type.upper() in ['CONTOUR', 'MESH', 'CONTOURLINE'], \
      'Invalid plot type choice - must be either contour, mesh or contourline'
    
    LR = param.L_R * 1e-3
#     if not plot_name:
#         plot_name=f'{canyon_choice}_canyonwidth={canyon_width*LR:.0f}km_\
# canyondepth={canyon_depth*param.H_D:.0f}m_\
# canyonlength={canyon_length*LR:.0f}km_{plot_type}_smooth={smooth}'
    x0, xN, y0, yN = bbox
    res_x, res_y = (xN - x0)/500, (yN - y0)/500
    x, y = np.arange(x0, xN+res_x, res_x), np.arange(y0, yN+res_y, res_y)
    X, Y = np.meshgrid(x, y)

    slope_topography, canyon_topography = coastal_topography(
        param,
        slope_choice,
        canyon_choice,
        coastal_shelf_depth,
        coastal_shelf_width,
        coastal_lengthscale,
        canyon_length,
        canyon_width,
        canyon_foot_depth,
        smooth=smooth,
        compare_filtering=True)
    
    from ppp.Plots import plot_setup, save_plot
    import matplotlib.pyplot as pt
    fig, ax = plot_setup("Cross-shore (km)",
                         "Bathymetry (m)", scale=.65)
    # ax.plot(y, -slope_topography(y), "k", label='Slope')
    # ax.plot(y, -canyon_topography(0, y), "r", label='Valley')
    for eps in canyon_width * np.linspace(0, .5, 6):
        ax.plot(y, -canyon_topography(eps, y), label=f'Canyon Wall: x={eps * param.L_R * 1e-3 :.2f}')
    ax.set_ylim([-1.1 , .1])
    ax.legend()
    pt.show()
    
    
    
    # H = param.H_D * canyon_topography(X, Y)
    # h_x, h_y = grad_function(H, res_y*param.L_R, res_x*param.L_R)
    
    # from ppp.Plots import plot_setup
    # import matplotlib.pyplot as pt
    # for h in [h_x, h_y]:
    #     fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
    #     c = ax.matshow(
    #         h,
    #         cmap="seismic",
    #         vmax=np.nanmax(h),
    #         vmin=-np.nanmax(h),
    #         extent=[x0*param.L_R*1e-3, xN*param.L_R*1e-3,
    #                 y0*param.L_R*1e-3, yN*param.L_R*1e-3],
    #         aspect="auto",
    #         origin="lower",
    #     )
    #     fig.colorbar(c, ax=ax)
    #     pt.show()

    # if plot_type.upper()=='CONTOURLINE':
    #     fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
    #     levels = np.linspace(200.001, 3999.999, 11)
    #     CS  = ax.contour(X*param.L_R*1e-3, Y*param.L_R*1e-3,
    #                      H, levels, cmap='jet_r')
    #     ax.clabel(CS, inline=1, fontsize=14)
        
    # elif plot_type.upper() == 'MESH':
    #     import matplotlib.pyplot as pt
    #     from mpl_toolkits.mplot3d import axes3d
    #     fig = pt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_xlabel('Along-shore (km)')
    #     ax.set_zticks(np.linspace(-4000, 0, 5))
    #     ax.set_ylabel('Cross-shore (km)')
    #     ax.zaxis.set_rotate_label(False)
    #     ax.set_zlabel('Bathymetry (m)', rotation=90,)# labelpad=10)
        
    #     ax.plot_surface(X*param.L_R*1e-3, Y*param.L_R*1e-3,
    #                       -H, rstride=5, cmap='YlGnBu_r')
    #     ax.plot_wireframe(X*param.L_R*1e-3, Y*param.L_R*1e-3, -H,
    #                       rstride=10, cstride=10,
    #                       color='black', linewidth=.25)
    #     ax.view_init(20, 65)
    #     fig.tight_layout()
    
    # else:
    #     fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
    #     c = ax.matshow(
    #         H,
    #         cmap="Blues",
    #         vmax=np.nanmax(H),
    #         vmin=0,
    #         extent=[x0*param.L_R*1e-3, xN*param.L_R*1e-3,
    #                 y0*param.L_R*1e-3, yN*param.L_R*1e-3],
    #         aspect="auto",
    #         origin="lower",
    #     )
    #     fig.colorbar(c, ax=ax)

    # if plot_name:
    #     from ppp.Plots import save_plot
    #     if plot_name.upper() == 'SAVE':
    #         plot_name = f'Canyon_{plot_type}'
    #     save_plot(fig, ax, plot_name, folder_name='Topographic Profiles')
        
    # else:
    #     import matplotlib.pyplot as pt
    #     pt.show()

if __name__ == "__main__":
    h_min, h_max = 5e-4, 5e-2 #Minimum and maximum edge-size values for mesh

    from barotropicSWEs.Configuration import configure
    param = configure.main()
    param.H_D = 4000
    param.H_C = 200
    param.c = np.sqrt(param.g * param.H_D)
    param.f, param.ω = 1e-4, 1.4e-4
    param.L_R = param.c/abs(param.f)
    param.L_C, param.L_S = 100e3, 50e3
    canyon_width = 15e3/param.L_R
    canyon_depth = 3000 / param.H_D
    canyon_length = 60e3 / param.L_R
    
    domain_width, domain_length = 50e3/param.L_R, 200e3/param.L_R
    coastal_lengthscale_ = (param.L_C + param.L_S)/param.L_R # Non-dimensional coastal lengthscale (shelf + slope)

    bbox = (-domain_width/2, domain_width/2, 0, domain_length)
    param.bbox = bbox
    param.resolution = .125 * 1e3/param.L_R

    for plot_type_ in ['Contour']: #'Mesh', 
        # for canyon_length_ in np.linspace(0, 120, 11)[-3:]*1e3/param.L_R:
        for smooth_ in [False]:
            plot_topography(bbox, param,
                            slope_choice='Cos_Squared',
                            canyon_choice='V-Shape',
                            coastal_lengthscale=coastal_lengthscale_,
                            coastal_shelf_depth=param.H_C/param.H_D,
                            coastal_shelf_width=param.L_C/param.L_R,
                            canyon_length=canyon_length,
                            canyon_foot_depth=canyon_depth,
                            canyon_width=canyon_width,
                            smooth=smooth_,
                            plot_name='',
                            plot_type=plot_type_)
