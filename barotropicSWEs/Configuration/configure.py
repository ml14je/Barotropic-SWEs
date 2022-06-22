#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Tue Dec  7 17:11:28 2021

"""
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='DG-FEM Barotropic Configuration'
        )

    parser.add_argument(
        '--order',
        type=int,
        default=2,
        help='Order of local polynomial test function in DG-FEM numerics.')
    
    parser.add_argument(
        '--domain',
        type=float,
        default=.1,
        help='Domain lengthscale')

    parser.add_argument(
        '--HD',
        type=float,
        default=4000.0,
        help='Depth of ocean beyond continental shelf.')

    parser.add_argument(
        '--HC',
        type=float,
        default=200.0,
        help='Depth of ocean along continental shelf.')

    parser.add_argument(
        '--hmin',
        type=float,
        default=5e-4,
        help='Minimum edge size of mesh.')

    parser.add_argument(
        '--hmax',
        type=float,
        default=5e-2,
        help='Maximum edge size of mesh.')

    parser.add_argument(
        '--coastal_lengthscale',
        type=float,
        default=150.0,
        help='Coastal lengthscale (shelf plus slope) in Rossby radii.')
    
    parser.add_argument(
        '--shelf_width',
        type=float,
        default=100.0,
        help='Shelf width in Rossby radii.')
    
    parser.add_argument(
        '--canyon_width',
        type=float,
        default=15.0,
        help='Canyon width in km.')
    
    parser.add_argument(
        '--canyon_length',
        type=float,
        default=60.0,
        help='Canyon length in km.')

    parser.add_argument(
        '--canyon_depth',
        type=float,
        default=2000.0,
        help='Depth of canyon at foot in m.')

    parser.add_argument(
        '--wave_frequency',
        type=float,
        default=1.4e-4,
        help='Forcing wave frequency. Default is semi-diurnal.')

    parser.add_argument(
        '--coriolis',
        type=float,
        default=1e-4,
        help='Local Coriolis coefficient. Default is typical of mid-latitude.')
    
    parser.add_argument(
        '--parameter',
        type=str,
        default='depth',
        choices=['length', 'depth', 'width'],
        help='Parameter for parameter sweep.')
    
    parser.add_argument(
        '--min_density',
        type=float,
        default=1026,
        help='Minimum density. Density of upper layer in two-layer model.')
    
    parser.add_argument(
        '--max_density',
        type=float,
        default=1036,
        help='Maximum density. Density of lower layer in two-layer model.')
    
    parser.add_argument(
        '--mean_density',
        type=float,
        default=1035.2,
        help='Mean density used as a reference for barotropic.')
    
    parser.add_argument(
        '--nbr_workers',
        type=int,
        default=1,
        help='Number of workers for parallel processing.')

    args = parser.parse_args()
    
    from math import sqrt
    import os
    from barotropicSWEs.Configuration import namelist

    param = namelist.Namelist() #object class holding parameter values as attributes
    param.Ω = 7.292e-5 #Angular speed of Earth's rotation (rad/s)
    param.ω = args.wave_frequency #Wave-frequency (s^{-1})
    param.f = args.coriolis #Coriolis parameter (s^{-1})
    param.ω_f = sqrt(param.ω**2-param.f**2)
    param.H_D = args.HD #Deep-sea vertical lengthscale (m)
    param.H_C = args.HC #Coastal vertical lengthscale (m)
    param.H_pyc = 150 #Depth of pycnocline (m)
    param.k = 7e-7 #Along-shore wavenumber (m^{-1})
    param.c = sqrt(param.g*param.H_D) #Charactertistic velocity scale (m/s)
    param.L_R = param.c/param.f #Rossby Radius of deformation (m)
    param.L_C = args.shelf_width * 1e3 #Horizontal coastal shelf lengthscale (m)
    param.L_S = args.coastal_lengthscale * 1e3 - param.L_C #Horizontal coastal slope lengthscale (m)
    param.r = 5e-2*param.f  #Damping coefficient (s^{-1})
    param.N = 5e-3 #Brunt-Väsälä frequency for stratification (s^-1)

    param.Ly = 2 * param.L_R #Off-shore domain size (m)
    param.Nx, param.Ny = 50, 50 #Grid-points in along-shore and cross-shore
    param.Nz = 256 #Grid-Points in vertical
    param.modes = 5 #Number of modes in vertical
    param.longitude, param.latitude = -40, 40 #Co-ordinates for observation data
    param.topography = 'Uniform' #Topography profile

    param.grid_type = 'C' #Available: ['A', 'C']
    param.horizontal_grid = 'Chebyshev' #Available: ['Chebyshev', 'Uniform']

    param.plot_scale = 1  #scale of plots w.r.t screen
    param.varplot = 'all' #Available: ['all', 'u', 'v', 'eta', 'phi']
    param.cmap = 'seismic' #Colour map for plots
    param.save_plot =  True #Save plot figures?
    param.print_info = False #Print Information relation to system parameters?

    param.ρ_max = args.max_density # maximum density
    param.ρ_min = args.min_density #minimum density
    param.reduced_gravity = param.g*((param.ρ_max-param.ρ_min)/param.ρ_max) #reduced-gravity felt at interface of two-layer fluid medium
    param.c_inf = np.sqrt(param.reduced_gravity * param.H_pyc)/param.c #non-dimensional baroclinic modal wave scale
    param.ρ_ref = args.mean_density
    param.boussinesq = True #Apply Boussinesq approximation?

    param.stratification = 'Two-Layer' #Stratification profile
    param.method = 'Multi-Layer'

    param.Nt = 1e2
    param.fps = 20
    param.repeat = 3

    from sympy import exp
    from sympy.abc import y
    param.eq_tidal_disp = 1.25
    param.coeff = 0.5*param.eq_tidal_disp*param.g
    param.body_force = param.coeff*exp(-y)
    param.wd = os.path.dirname(__file__)
    param.data_dir = os.path.join(param.wd, '..', '..', 'Ocean Dataset')
    param.folder_dir = os.path.join(
            param.topography, 'Numerics'
            )
    
    param.resolution = .125 * 1e3/param.L_R
    param.canyon_depth = args.canyon_depth
    param.canyon_length = args.canyon_length
    param.canyon_width = args.canyon_width
    param.alpha = 0.35
    param.beta = 0.5

    param.domain = args.domain
    param.bbox = (-param.domain/2, param.domain/2, 0, param.domain)
    param.bbox_dimensional = tuple([x * 1e-3 * param.L_R for x in param.bbox])
    param.domain_size = (param.domain, param.domain)
    param.order = args.order
    param.hmin, param.hmax = args.hmin, args.hmax
    param.parameter = args.parameter
    param.nbr_workers = args.nbr_workers
    return param
