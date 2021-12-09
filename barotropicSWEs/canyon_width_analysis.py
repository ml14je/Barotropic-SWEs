#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sun Aug 22 12:27:53 2021

"""
import numpy as np

def yield_data(swes, sols, bbox_plot=(-.025, .025, 0, .05)):
    """
    The will create a dictionary of variables plotted on a regular mesh ready
    for post-processing.

    Parameters
    ----------
    swes : Barotropic
        A Barotropic class which holds our variable values.
    sols : numpy.array
        Spatial data corrsponding to Barotropic model.
    bbox_plot : tuple, optional
        Boundary box within which we desire. The default is
        (-.025, .025, 0, .05), which corresponds to, using default values,
        an approximate 100km x 100km domain.

    Returns
    -------
    dict : dict
        A dictionary of variable spatial data interpolated on a regular mesh
        of our new boundary, bbox.
    grid : tuple
        A tuple of spatial grids.

    """
    from process_slns import gridded_data

    vals, grid = gridded_data(swes, sols, bbox_plot)
    labels = ['u', 'v', 'eta', 'Qx', 'Qy', 'vorticity', 'ugradh',
              'uh_x', 'vh_y', 'KEx', 'KEy', 'KE', 'PE']

    return {label:val for label,val in zip(labels, vals)}, grid

def canyon_analysis(bbox, param, order=3, h_min=1e-3, h_max=5e-3,
             wave_frequency=1.4, wavenumber=1.4, rayleigh_friction=0,
             coastal_lengthscale=0.03, canyon_widths=[0, 1e-3, 5e-3, 1e-2],
             potential_forcing=False, scheme='Lax-Friedrichs',
             norm='L1', plot_norms=False):
    """
    Produces a dictionary of variable values along a cross-section for
    different canyon widths. It will also save these dictionaries for future
    analyses.

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
    norm : str, optional
        The norm to map our two-dimensional spatial data to a single value. For
        example, L1, L2 or L_inf. The default is L1, which is essentially an
        equally-weighted average.
    plot_norms : bool, optional
        Whether to save individual norm plots. The default is False.

    Returns
    -------
    sect_list : dict
        A dictionary of variables and thir values along the cross-section
        for different canyon widths.

    """
    
    from SWEs import startup, boundary_value_problem
    from ppp.File_Management import dir_assurer, file_exist
    import pickle
    
    ω, k, r = wave_frequency, wavenumber, rayleigh_friction
    λ = coastal_lengthscale
    dir_assurer('Norms')
    norm = norm.upper()
    assert norm in ['L1', 'L2', 'LINF'], \
           "Choice of norm not supported. Choices are 'L1', 'L2' and 'LINF'."
    norm_func = {'L1': L1_norm, 'L2': L2_norm, 'LINF': Linf_norm}[norm]
    norm_list = []
    for background_flow_ in ['KELVIN', 'CROSSSHORE']:
        norms = {}
        file_dir = f'DomainSize={bbox[-1]*param.L_R*1e-3:.0f}_{norm}\
 _{background_flow_}_ω={ω:.1f}'
        norm_data_name = file_dir + '.pkl'
        if not file_exist(f'Norms/{norm_data_name}'):
            for w_ in canyon_widths:
                print(f'Canyon width: {w_*param.L_R*1e-3:.2f}km')
                swes, φ, swe_dir = startup(
                    bbox,
                    param,
                    h_min,
                    h_max,
                    order,
                    mesh_name="",
                    canyon_width=w_,
                    plot_domain=False,
                    boundary_conditions="Specified",
                    scheme=scheme,
                    potential_forcing=potential_forcing,
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
                    file_name=swe_dir,
                    animate=False,
                )

                val_dic, grid = yield_data(swes, sols)
                norm_dic = {}
                xg, yg = grid
                normalisation = norm_func(np.ones((len(xg), len(yg))))
                for key in val_dic.keys():
                    data = val_dic[key]
                    norm_val = norm_func(data)
                    assert not np.isnan(norm_val), "Norm value is NaN"
                    norm_dic[key] = norm_val/normalisation

                norms[w_] = norm_dic
            with open(f'Norms/{norm_data_name}', 'wb') as outp:
                pickle.dump(norms, outp, pickle.HIGHEST_PROTOCOL)

        else:
            with open(f'Norms/{norm_data_name}', 'rb') as inp:
                norms = pickle.load(inp)

        norm_list.append(norms)

        if plot_norms:
            from ppp.Plots import plot_setup, save_plot
            dir_assurer(f'Norms/{file_dir}')
            labels = ['u', 'v', 'eta', 'Qx', 'Qy', 'vorticity', 'ugradh',
                  'uh_x', 'vh_y', 'KEx', 'KEy', 'KE', 'PE']

            for label in labels:
                fig, ax = plot_setup('Canyon Width ($\\rm{km}$)',
                                     f'{norm}-Norm')
                ax.plot(w_vals[1:] * param.L_R/1e3, [norms[w_][label] for \
                                                     w_ in w_vals[1:]], '-',
                        c='k')
                ax.scatter(w_vals[1:] * param.L_R/1e3, [norms[w_][label] for \
                                                        w_ in w_vals[1:]],
                           marker='x', c='r')
                ax.axhline(y=norms[0][label], c='k', ls=':')
                save_plot(fig, ax, label, folder_name=f'Norms/{file_dir}')

    return norm_list

def Lp_norm(u, p):
    """
    Generic L-p norm.

    Parameters
    ----------
    u : numpy.array
        Array of valus on which to carry out the normalisation.
    p : float
        p-value for norm.

    Returns
    -------
    float
        The L-p norm of u.

    """
    return np.sum(abs(u) ** p) ** (1/p)

def Linf_norm(u):
    """
    Returns the infinity-norm, which is essentially the maximum value of input
    u.

    Parameters
    ----------
    u : numpy.array
        Vector array on which to perform the infinity norm.

    Returns
    -------
    float
        The infinity-norm of u.

    """
    return np.nanmax(np.abs(u))

def L1_norm(u):
    """
    Returns the L1-norm of input vector u.

    Parameters
    ----------
    u : numpy.array
        Vector array on which to perform the L1 norm.

    Returns
    -------
    float
        The L1-norm of u.

    """
    return Lp_norm(u, 1)

def L2_norm(u):
    """
    Returns the L2-norm of input vector u.

    Parameters
    ----------
    u : numpy.array
        Vector array on which to perform the L2 norm.

    Returns
    -------
    float
        The L2-norm of u.

    """
    return Lp_norm(u, 2)

if __name__ == "__main__":
    h_min, h_max = 5e-4, 5e-2
    order=5
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
    w_vals = np.linspace(1e-3, 1e-2, 19) #[2::4]
    w_vals = np.insert(w_vals, 0, 0)

    sols_dict = {}
    for domain_width in [.05, .1, .15]:
        bbox = (-domain_width/2, domain_width/2, 0, domain_width)
        sols_dict[domain_width] = {}
        print(f'Domain size: {domain_width*param.L_R*1e-3:.0f}km')
        for norm_ in ['L1', 'L2', 'LINF']:
            dictionary = canyon_analysis(bbox, param, order, h_min, h_max,
                         wave_frequency=ω, wavenumber=k, rayleigh_friction=0,
                         coastal_lengthscale=λ, canyon_widths=w_vals,
                         norm=norm_, plot_norms=False)
            sols_dict[domain_width][norm_] = dictionary

    from ppp.Plots import subplots, set_axis, save_plot
    from ppp.File_Management import dir_assurer


    dir_assurer('Norms/Summary_Analysis/CrossShore')
    units = ['m/s', 'm/s', 'm', 'm^2/s', 'm^2/s', 's^{-1}', 'm/s',
              'm/s', 'm/s', 'kJ', 'kJ', 'kJ', 'kJ']
    titles = ['u', 'v', '\\eta', 'h\\,u', 'h\\,v', '\\partial_xv-\\partial_yu',
              '\\mathbf{u}\\cdot\\nabla h', 'u\\,h_x', 'v\\,h_y',
              'Along-shore Kinetic', 'Cross-shore Kinetic',
              'Total Kinetic Energy', 'Total Potential Energy']

    for j, forcing_ in enumerate(['Kelvin', 'CrossShore']):
        dir_assurer(f'Norms/Summary_Analysis/{forcing_}')

        for k, label in enumerate(['u', 'v', 'eta', 'Qx', 'Qy', 'vorticity',
                                   'ugradh', 'uh_x', 'vh_y', 'KEx', 'KEy',
                                   'KE', 'PE']):
            fig, axis = subplots(1, 3, y_share=True)
            if k in [9, 10, 11, 12]:
                prefactor = 1e-3
                title_ = titles[k]

            else:
                prefactor = 1
                title_ = f'${titles[k]}$'

            for i, norm_ in enumerate(['L1', 'L2', 'LINF']):
                ax = axis[i]
                lines_, labels_ = [], []
                y_title = f'{title_} ($\\rm{{{units[k]}}}$)' if i==0 else ''
                set_axis(ax, 'Canyon Width ($\\rm{km}$)', y_title,
                          title=f'{norm_}-Norm', scale=.85)
                for domain_width in [.05, .1, .15]:
                    Ldomain = param.L_R * domain_width * 1e-3
                    Ldomain = round(Ldomain, -int(np.log10(Ldomain)))
                    L10 = round(int(np.log10(Ldomain)))
                    vals = [sols_dict[domain_width][norm_][j][w_][label] * \
                            prefactor for w_ in w_vals]
                    l, = ax.plot(w_vals[1:] * param.L_R/1e3, vals[1:], '-')
                    ax.scatter(w_vals[1:] * param.L_R/1e3, vals[1:],
                               marker='x', c='r', linewidth=.5)
                    lines_.append(l)
                    labels_.append(f'$\\approx{Ldomain:.0f}$ ($\\rm{{km}}$)')

            axis[1].legend(handles = lines_, labels=labels_,
                           loc='upper center', title='Domain Size:',
                           bbox_to_anchor=(0.5, -0.2), fancybox=False,
                           shadow=False, ncol=3, title_fontsize=18,
                           fontsize=16)
            save_plot(fig, axis, f'{label}_norms',
                      folder_name=f'Norms/Summary_Analysis/{forcing_}')
