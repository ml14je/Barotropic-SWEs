#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Mon Mar 21 10:56:21 2022

"""
import numpy as np

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Mon Nov  30 15:33:46 2021

"""


def zhang_duda_udovydchenkov_2014(X, Y,
                                  y_i=0,
                                  coastal_shelf_width=100,
                                  coastal_slope_width=50,
                                  shelf_depth=200,
                                  ocean_depth=4000,
                                  canyon_length=35.0,
                                  canyon_width=15.0):

    y_c = coastal_shelf_width - canyon_length * \
        np.exp(-(X**2)/(2 * ((canyon_width/2)**2)))

    return -((ocean_depth+shelf_depth)/2 -
             (ocean_depth - shelf_depth)/2 * np.tanh((y_c - Y)/(coastal_slope_width/2)))


@np.vectorize
def baines_1983a(X, Y,
                 y_i=0,
                 coastal_shelf_width=100.0,
                 coastal_slope_width=50.0,
                 shelf_depth=200,
                 ocean_depth=4000,
                 canyon_length=35.0,
                 canyon_width=15.0):

    if abs(X) < canyon_width/2:
        y_c = coastal_shelf_width - canyon_length

    else:
        y_c = coastal_shelf_width

    if Y <= y_c:
        return -shelf_depth

    else:
        return np.maximum(-ocean_depth,
                          -(shelf_depth +
                            (ocean_depth-shelf_depth)*(Y - y_c) /
                            coastal_slope_width)
                          )

@np.vectorize
def baines_1983b(X, Y,
                 coastal_shelf_width=100,
                 coastal_slope_width=50.0,
                 shelf_depth=200,
                 ocean_depth=4000,
                 canyon_length=35.0,
                 canyon_width=13.0):

    delta_canyon_width = 2.0

    if 2* abs(X) < canyon_width:
        y_c = coastal_shelf_width - canyon_length
        l_c = coastal_slope_width

    elif 2 * abs(X) > canyon_width + delta_canyon_width:
        y_c = coastal_shelf_width
        l_c = coastal_slope_width
        
    else:
        y_c = coastal_shelf_width
        l_c =  coastal_slope_width - canyon_length + \
            ( canyon_length) * \
                (2*abs(X) - canyon_width)/delta_canyon_width

    if Y <= y_c:
        return -shelf_depth

    else:
        return np.maximum(-ocean_depth,
                          -(shelf_depth +
                            (ocean_depth-shelf_depth)*(Y - y_c) /
                            l_c)
                          )

@np.vectorize
def nazarian_2017a(X, Y,
                 coastal_shelf_width=100,
                 coastal_slope_width=50.0,
                 shelf_depth=200,
                 ocean_depth=4000,
                 canyon_length=85.0,
                 canyon_width=15.0):
    



    y_c = coastal_shelf_width
    l_c = coastal_slope_width

    if Y <= y_c:
        H =  -shelf_depth

    else:
        H =  np.maximum(-ocean_depth,
                          -(shelf_depth +
                            (ocean_depth-shelf_depth)*(Y - y_c) /
                            l_c)
                          )
    
    ΔL, w = canyon_length, canyon_width
    L0, λ = coastal_shelf_width, coastal_shelf_width + coastal_slope_width
    λ0 = coastal_shelf_width + coastal_slope_width
    h = shelf_depth
    h2 = ocean_depth
    
    if w > 0:
        p0, p1 = np.array([0, λ0, h2]), np.array([0, λ0-ΔL, h])
        p2, p3 = np.array([-w/2, L0, h]), np.array([w/2, L0, h])
        # print(p0, p1, p2, p3)
        
        N1, N2 = p2 - p0, p2 - p1
        n = np.array([N1[1] * N2[2] - N1[2] * N2[1],
                      N1[2] * N2[0] - N1[0] * N2[1],
                      N1[0] * N2[1] - N1[1] * N2[0]])
    
        if (((λ0 >= Y) & (Y >= L0))  & ((-w/2 * (1 -(Y-L0)/(λ0-L0)) < X) & (X <= 0)) | \
                        ((L0 >= Y) & (Y >= λ0-ΔL))  & ((-w/2 * (0 + (Y-λ0+ΔL)/(L0-λ0+ΔL)) < X) & (X <= 0))):
            # print(-((n[0] * (X - p0[0]) + n[1] * (Y - p0[1]) - n[2] * p0[2])/n[2]))
            H = min(((n[0] * (X - p0[0]) + n[1] * (Y - p0[1]) - n[2] * p0[2])/n[2]), H)
            
        N1, N2 = p3 - p0, p3 - p1
        n = np.array([N1[1] * N2[2] - N1[2] * N2[1],
                      N1[2] * N2[0] - N1[0] * N2[1],
                      N1[0] * N2[1] - N1[1] * N2[0]])  
        
        if (((λ0 >= Y) & (Y >= L0))  & ((w/2 * (1 -(Y-L0)/(λ0-L0)) > X) & (X >= 0)) | \
                        ((L0 >= Y) & (Y >= λ0-ΔL))  & ((w/2 * (0 +(Y-λ0+ΔL)/(L0-λ0+ΔL)) > X) & (X >= 0))):
            H = min(((n[0] * (X - p0[0]) + n[1] * (Y - p0[1]) - n[2] * p0[2])/n[2]), H)
        
    return H
    
@np.vectorize
def nazarian_2017b(X, Y,
                 coastal_shelf_width=100,
                 coastal_slope_width=50.0,
                 shelf_depth=200,
                 ocean_depth=4000,
                 canyon_length=35.0,
                 canyon_width=15.0):
    
    y_c = coastal_shelf_width
    l_c = coastal_slope_width

    if Y <= y_c:
        H =  -shelf_depth

    else:
        H =  np.maximum(-ocean_depth,
                          -(shelf_depth +
                            (ocean_depth-shelf_depth)*(Y - y_c) /
                            l_c)
                          )
        
    if 2 * abs(X) < canyon_width and \
        Y > coastal_shelf_width + coastal_slope_width + \
            (canyon_length + coastal_slope_width) * \
            (2 * abs(X) - canyon_width)/canyon_width:
        H = -ocean_depth
        
    return H

def coastal_topography(param,
                       topography_choice='Nazarian_2017a'):

    if topography_choice.upper() == 'ZHANG_DUDA_UDOVYDCHENKOV_2014':
        def topography_function(
            x, y): return zhang_duda_udovydchenkov_2014(x, y)

    elif topography_choice.upper() == 'BAINES_1983A':
        def topography_function(x, y): return baines_1983a(x, y)

    elif topography_choice.upper() == 'BAINES_1983B':
        def topography_function(x, y): return baines_1983b(x, y)
        
    elif topography_choice.upper() == 'NAZARIAN_2017A':
        def topography_function(x, y): return nazarian_2017a(x, y)
        
    elif topography_choice.upper() == 'NAZARIAN_2017B':
        def topography_function(x, y): return nazarian_2017b(x, y)

    else:
        raise ValueError("Incorrect choice of slope function.")

    return topography_function


def plot_topography(bbox, param,
                    topography_choice='Zhang_Duda_Udovydchenkov_2014',
                    coastal_lengthscale=.03,
                    coastal_shelf_width=.02,
                    coastal_shelf_depth=.05,
                    canyon_length=.01,
                    canyon_width=5e-3,
                    canyon_depth=1.0,
                    plot_name='',
                    plot_type='CONTOUR'):

    assert plot_type.upper() in ['CONTOUR', 'MESH', 'CONTOURLINE'], \
        'Invalid plot type choice - must be either contour, mesh or contourline'

    LR = param.L_R * 1e-3
    if not plot_name:
        plot_name = f'{topography_choice}'
    x0, xN, y0, yN = bbox

    x = np.linspace(x0, xN, 501) * param.L_R * 1e-3
    y = np.linspace(y0, yN, 501) * param.L_R * 1e-3
    X, Y = np.meshgrid(x, y)

    topography_function = coastal_topography(
        param, topography_choice,
        )

    H = topography_function(X, Y)

    from ppp.Plots import plot_setup

    if plot_type.upper() == 'CONTOURLINE':
        fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
        levels = np.linspace(200.001, 3999.999, 11)
        CS = ax.contour(X*param.L_R*1e-3, Y*param.L_R*1e-3,
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
        ax.set_zlabel('Bathymetry (m)', rotation=90,)  # labelpad=10)

        ax.plot_surface(X, Y,
                        H, rstride=5, cmap='YlGnBu_r')
        ax.plot_wireframe(X, Y, H,
                          rstride=10, cstride=10,
                          color='black', linewidth=.25)
        ax.view_init(20, 65)
        fig.tight_layout()

    else:
        fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
        c = ax.matshow(
            H,
            cmap="Blues",
            vmax=np.nanmin(H),
            vmin=0,
            extent=[x0*param.L_R*1e-3, xN*param.L_R*1e-3,
                    y0*param.L_R*1e-3, yN*param.L_R*1e-3],
            aspect="auto",
            origin="lower",
        )
        ax.plot([2.5, 5], [100, 150], 'rx')
        fig.colorbar(c, ax=ax)

    if plot_name:
        from ppp.Plots import save_plot
        if plot_name.upper() == 'SAVE':
            plot_name = f'Canyon_{plot_type}'
        save_plot(fig, ax, plot_name, folder_name='Topographic Profiles')

    else:
        import matplotlib.pyplot as pt
        pt.show()


if __name__ == "__main__":
    h_min, h_max = 5e-4, 5e-2  # Minimum and maximum edge-size values for mesh

    from barotropicSWEs.Configuration import configure
    param = configure.main()
    param.H_D = 4000
    param.H_C = 200
    param.c = np.sqrt(param.g * param.H_D)
    param.f, param.ω = 1e-4, 1.4e-4
    param.L_R = param.c/abs(param.f)
    canyon_width_ = 15e3/param.L_R
    # Non-dimensional domain width and length
    domain_width, domain_length = 30e3/param.L_R, 200e3/param.L_R
    # Non-dimensional coastal lengthscale (shelf + slope)
    coastal_lengthscale_ = .075

    bbox = (-domain_width/2, domain_width/2, 0, domain_length)
    
    for topography_choice_ in ['Zhang_Duda_Udovydchenkov_2014', 'Baines_1983a',
                                'Baines_1983b', 'Nazarian_2017a',
                                'Nazarian_2017b'
                                ]:
        plot_topography(bbox, param,
                        topography_choice=topography_choice_,
                        coastal_lengthscale=150e3/param.L_R,
                        coastal_shelf_depth=param.H_C/param.H_D,
                        coastal_shelf_width=50e3/param.L_R,
                        canyon_length=60e3/param.L_R,
                        canyon_depth=.5,
                        canyon_width=canyon_width_,
                        plot_type='MESH')
