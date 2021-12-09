#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sun Aug 22 12:27:53 2021

"""
import numpy as np

def canyon_analysis(bbox, param, order=3, h_min=1e-3, h_max=5e-3,
             wave_frequency=1.4, wavenumber=1.4, rayleigh_friction=0,
             coastal_lengthscale=0.03, canyon_widths=[0, 1e-3, 5e-3, 1e-2],
             potential_forcing=False, scheme='Lax-Friedrichs',
             variable='x', value=0, plot_sects=False):
    """
    Produces a dictionary of variable values along a cross-section for different canyon
    widths. It will also save these dictionaries for future analyses.

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
    variable : str, optional
        The spatial variable along which to present a cross-sectional plot.
        The default variable is 'x'.
    value : float, optional
        The value of the variable along which to present the cross-sectional
        plot. The default value is 0.
    plot_sects : bool, optional
        Whether to save individual cross-sectional plots. The default is False.

    Returns
    -------
    sect_list : dict
        A dictionary of variables and thir values along the cross-section
        for different canyon widths.

    """
    
    from ppp.File_Management import dir_assurer, file_exist
    import pickle
    from SWEs import startup, boundary_value_problem
    from canyon_width_analysis import yield_data

    ω, k, r = wave_frequency, wavenumber, rayleigh_friction
    λ = coastal_lengthscale

    assert variable in ['x', 'y'], \
           "Variable choice options are either 'x' (along-shore) or 'y' (cross-shore)."

    if variable == 'x':
        assert (-.025 <= value) and (.025 >= value), \
                  "Your choice of cross-sectional coordinate lies beyond the domain"

    if variable == 'y':
        assert (0 <= value) and (.05 >= value), \
                  "Your choice of along-sectional coordinate lies beyond the domain"

    units = ['m/s', 'm/s', 'm', 'm^2/s', 'm^2/s', 's^{-1}', 'm/s',
             'm/s', 'm/s', 'kJ', 'kJ', 'kJ', 'kJ']
    titles = ['u', 'v', '\\eta', 'h\\,u', 'h\\,v', '\\partial_xv-\\partial_yu', '\\mathbf{u}\\cdot\\nabla h',
                'u\\,h_x', 'v\\,h_y', 'Along-shore Kinetic', 'Cross-shore Kinetic', 'Total Kinetic Energy',
                'Total Potential Energy']

    sect_list = []
    value *= param.L_R
    dir_assurer('SectionalData')
    for background_flow_ in ['KELVIN', 'CROSSSHORE']:
        sects = {}
        file_dir = f'DomainSize={bbox[-1]*param.L_R*1e-3:.0f}\
_{background_flow_}_ω={ω:.1f}_{variable}={1e-3*value:.0f}'
        sectional_data_name = file_dir + '.pkl'
        if not file_exist(f'SectionalData/{sectional_data_name}'):
            for w_ in canyon_widths:
                print(f'Canyon width: {w_*param.L_R*1e-3:.2f}km')
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
                    file_name=file_dir,
                    animate=False,
                )

                val_dic, grid = yield_data(swes, sols)

                sect_dic = {}
                xg, yg = grid
                sects['grid'] = xg if variable == 'y' else yg
                for key in val_dic.keys():
                    data = val_dic[key]
                    sectional_data = extract_data(data, variable, value, xg, yg)
                    sect_dic[key] = sectional_data

                sects[w_] = sect_dic
            with open(f'SectionalData/{sectional_data_name}', 'wb') as outp:
                pickle.dump(sects, outp, pickle.HIGHEST_PROTOCOL)

        else:
            with open(f'SectionalData/{sectional_data_name}', 'rb') as inp:
                sects = pickle.load(inp)

        sect_list.append(sects)

        if plot_sects:
            dir_assurer(f'SectionalData/{file_dir}')
            import matplotlib
            matplotlib.use('Agg')
            from ppp.Plots import save_plot
            labels = ['u', 'v', 'eta', 'Qx', 'Qy', 'vorticity', 'ugradh',
                  'uh_x', 'vh_y', 'KEx', 'KEy', 'KE', 'PE']

            x_label = 'Along-shore (km)' if variable == 'y' else 'Cross-shore (km)'
            w_vals_ = canyon_widths[4::5]
            w_vals_ = np.insert(w_vals_, 0, 0)
            for i in range(len(labels)):
                if i not in [9, 10, 11, 12]:
                    title_ = f'${titles[i]}$'
                    prefactor=1

                else:
                    title_ = titles[i]
                    prefactor=1e-3

                fig, ax = plot_setup(x_label, f'{title_} ($\\rm{{{units[i]}}}$)')
                phase = np.angle(sects[0][labels[i]][np.argmax(abs(sects[0][labels[i]]))])
                ax.plot(sects['grid']*1e-3, prefactor*np.array([np.round(sects[w_][labels[i]] * \
                                                     np.exp(-1j*phase), 16) \
                                                     for w_ in w_vals_]).real.T, '-')
                ax.legend([f'{w*param.L_R*1e-3 :.0f}' for w in w_vals_], title='Canyon Width (km):',
                          fontsize=16, title_fontsize=18, loc=1)
                save_plot(fig, ax, labels[i], folder_name=f'SectionalData/{file_dir}')
                pt.close('all')

    return sect_list

def extract_data(data, variable, value, xg, yg):
    """
    Inteprolates variable mesh data onto a fine mesh along the cross-section
    variable=value.

    Parameters
    ----------
    data : numpy.array
        Values from our FEM mesh from which we wish to interpolate
        onto the finer regular mesh.
    variable : str, optional
        The spatial variable along which to present a cross-sectional plot.
    value : float, optional
        The value of the variable along which to present the cross-sectional
        plot.
    xg : numpy.array
        Fine along-shore mesh.
    yg : numpy.array
        Fine cross-shore mesh.

    Returns
    -------
    numpy.array
        An array of values interpolated from the FEM mesh onto the regular fine mesh, (xg, yg),
        along variable=value.

    """
    from scipy.interpolate import interp2d
    xg, yg = np.round(xg, 15), np.round(yg, 15)

    if variable == 'x':
        X, Y = value, yg

    else:
        X, Y = xg, value

    func_r = interp2d(xg, yg, data.real, kind='cubic')
    func_i = interp2d(xg, yg, data.imag, kind='cubic')

    return (func_r(X, Y) + 1j * func_i(X, Y)).flatten()

if __name__ == "__main__":
    h_min, h_max = 5e-4, 5e-2
    order = 5
    λ = 0.03

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
    w_vals_ = np.linspace(1e-3, 1e-2, 19)#[2::4]
    w_vals_ = np.insert(w_vals_, 0, 0)

    sols_dict = {}
    for domain_width in [.05, .1, .15]:
        bbox = (-domain_width/2, domain_width/2, 0, domain_width)
        sols_dict[domain_width] = {}
        print(f'Domain size: {domain_width*param.L_R*1e-3:.0f}km')
        for var_ in ['x', 'y']:
            if var_ == 'x':
                values_ = [-.025, 0, .025]
            else:
                values_ = [.01, .02, .03, .05]

            for value_ in values_:
                dictionary = canyon_analysis(bbox, param, order,
                             h_min, h_max, wave_frequency=ω, wavenumber=k, 
                             rayleigh_friction=0, coastal_lengthscale=λ,
                             canyon_widths=w_vals_, variable=var_,
                             value=value_, plot_sects=False)
                sols_dict[domain_width][f'{var_}={value_}'] = dictionary

    import matplotlib.pyplot as pt
    from ppp.Plots import plot_setup, subplots, set_axis, save_plot
    from ppp.File_Management import dir_assurer


    units = ['m/s', 'm/s', 'm', 'm^2/s', 'm^2/s', 's^{-1}', 'm/s',
             'm/s', 'm/s', 'kJ', 'kJ', 'kJ', 'kJ']
    titles = ['u', 'v', '\\eta', 'h\\,u', 'h\\,v', '\\partial_xv-\\partial_yu', '\\mathbf{u}\\cdot\\nabla h',
                'u\\,h_x', 'v\\,h_y', 'Along-shore Kinetic', 'Cross-shore Kinetic', 'Total Kinetic Energy',
                'Total Potential Energy']
    w_vals_ = np.array([0, 1e-3, 1e-2])
    L_vals_ = np.array([.05, .1, .15])
    labels_ = [f'{np.round(L*param.L_R*1e-3, -2):.0f}' for L in L_vals_]

    for j, forcing_ in enumerate(['Kelvin', 'CrossShore']):
        dir_assurer(f'SectionalData/Summary_Analysis/{forcing_}')
        for k, label in enumerate(['u', 'v', 'eta', 'Qx', 'Qy', 'vorticity', 'ugradh',
                                   'uh_x', 'vh_y', 'KEx', 'KEy', 'KE', 'PE']):
            for var_ in ['x', 'y']:
                if var_ == 'x':
                    values_ = [-.025, 0, .025]
                else:
                    values_ = [.01, .02, .03, .05]

                for value_ in values_:
                    x_label = 'Along-shore (km)' if var_ == 'y' else 'Cross-shore (km)'
                    fig, axis = subplots(1, 3, y_share=True)
                    fig.suptitle('Canyon Width (km)', fontsize=20)

                    if k in [9, 10, 11, 12]:
                        prefactor = 1e-3
                        title_ = titles[k]

                    else:
                        prefactor = 1
                        title_ = f'${titles[k]}$'

                    for i, w_ in enumerate(w_vals_):
                        vals = sols_dict[L_vals_[0]][f'{var_}={value_}'][j][w_][label]
                        phases = [np.angle(vals[np.argmax(np.abs(vals))]) for \
                                  L in L_vals_]
                        canyon_width = w_ * param.L_R * 1e-3
                        ax = axis[i]
                        y_title = f'{title_} ($\\rm{{{units[k]}}}$)' if i==0 else ''
                        set_axis(ax, x_label, y_title,
                                 title=f'{canyon_width:.0f}', scale=.85)
                        vals = np.array(
                             [sols_dict[L][f'{var_}={value_}'][j][w_][label]*\
                              prefactor*np.exp(-1j*phases[0]) for L in L_vals_]
                                       ).T
                        l1, l2, l3, = ax.plot(sols_dict[domain_width][f'{var_}=\
{value_}'][j]['grid']*1e-3, vals.real, '-')
                        handles_ = [l1, l2, l3]
                    axis[1].legend(handles = [l1, l2, l3], labels=labels_,
                                   loc='upper center', title='Domain Size:',
                                   bbox_to_anchor=(0.5, -0.2), fancybox=False,
                                   shadow=False, ncol=3,
                                   title_fontsize=18, fontsize=16)
                    save_plot(fig, axis,
                        f'{label}_{var_}={value_*param.L_R*1e-3:.0f}',
                        folder_name=f'SectionalData/Summary_Analysis/{forcing_}')
