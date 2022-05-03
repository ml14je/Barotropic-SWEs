#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Tue Feb 22 15:20:13 2022

"""
import numpy as np

def validation_cases(param, case="square", refinement="p", slope='GG07',
                     element_edge=None, default_order=None,
                     plot_scaling=False):
    fluxes = ["Central",
               "Upwind",
               "Penalty",
              "Lax-Friedrichs",
               "Alternating"
              ]
    default_order = 2 if default_order is None else default_order
    orders = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert case.upper() in ["NONROTATING", "ROTATING", "KELVIN"], \
        "Invalid test case selected."
    assert refinement.upper() in ["H", "P"], \
        "Invalid choice of refinement."

    if case.upper()=="NONROTATING":
        print(f"Conducting {refinement}-refinement for non-rotating \
eigenvalue problem")
        element_edge = .25 if element_edge is None else element_edge
        from barotropicSWEs.Tests import nonrotating
        if refinement.upper()=="P":
            for order_ in orders:
                nonrotating.square_eigenvalue_problem(param,
                                                      order=order_,
                                                      h_target=element_edge,
                                                      tolerance=.1,
                                                      numerical_fluxes=fluxes,
                                                      theta_value=.25,
                                                      verbose=False
                                                      )
        else:
            for h_ in element_edge * (1/2)**np.array([0, 1, 2, 3]):
                for order_ in [2, 3]:
                    nonrotating.square_eigenvalue_problem(param,
                                                          order=order_,
                                                          h_target=h_,
                                                          tolerance=.1,
                                                          numerical_fluxes=fluxes,
                                                          theta_value=.25,
                                                          error_base=2,
                                                          verbose=False
                                                          )
                
    elif case.upper()=="ROTATING":
        print(f"Conducting {refinement}-refinement for rotating \
eigenvalue problem")
        element_edge = .25 if element_edge is None else element_edge
        from barotropicSWEs.Tests import rotating
        if refinement.upper()=="P":
            for order_ in orders:
                rotating.circle_eigenvalue_problem(param,
                                                   order=order_,
                                                   h_target=element_edge,
                                                   numerical_fluxes=fluxes,
                                                   theta_value=.25,
                                                   verbose=False
                                                   )
        else:
            for h_ in element_edge * (1/2)**np.array([-1, 0, 1]):
                errors = rotating.circle_eigenvalue_problem(
                    param,
                    order=default_order,
                    h_target=h_,
                    numerical_fluxes=fluxes,
                    theta_value=.25,
                    verbose=False
                    )
                
    else:
        print(f"Conducting {refinement}-refinement for Kelvin \
boundary-value problem")
        element_edge = .05 if element_edge is None else element_edge
        from barotropicSWEs.Tests import kelvin
        if refinement.upper()=="P":
            plot_spatial_errors = False
            for length_ in [300/2e3, 600/2e3]:
                bbox = (-length_/2, length_/2, 0, length_)
                folder_name = f'BoundaryValue_Problem/h={element_edge:.2e}/\
Domain={length_*1e-3*param.L_R:.0f}km/Slope={slope}'
                error_norms = {}
                time_arr = np.zeros((len(orders), len(fluxes)))
                for order_ in orders:
                    times, error_norm = \
                        kelvin.kelvin_boundaryvalue_problem(
                            param,
                            bbox,
                            order=order_,
                            h_target=element_edge,
                            tolerance=.1,
                            slope_profile=slope,
                            numerical_fluxes=fluxes,
                            theta_value=.5,
                            error_base=10,
                            plot_exact=False,
                            plot_topography=False,
                            regular_solutions=True,
                            plot_spatial_error=True,
                            verbose=True
                            )
                    time_arr[order_ - 1] = times
                    error_norms[order_] = error_norm

                from ppp.Plots import plot_setup, save_plot
                import matplotlib.pyplot as pt
                fig, ax = plot_setup('Order', 'Time (s)', y_log=True, by=2, scale=.7)
                ax.plot(orders, time_arr, 'x-')
                

                if plot_scaling:
                    KK = 2 * (length_/element_edge)**2
                    Np = (orders + 1)*(orders + 2)/2
                    ax.plot(orders, (2**(-15)) * (KK * Np)**(3/2),
                            'k:')

                ax.set_xticks(orders)
                ax.set_xticklabels(orders)
                ax.legend(fluxes, fontsize=16)
                pt.show()
                save_plot(fig, ax, f'Time_L={length_*param.L_R*1e-3:.0f}km_h=\
{element_edge * param.L_R * 1e-3:.0f}km', folder_name=folder_name,
                          labels=fluxes)
                    
#                 x0, xN, y0, yN = bbox
            
                    
#                 L1_fig, L1_ax = plot_setup("Order", "Normalised $L^1$ Error",
#                                             y_log=True, by=2)
#                 L2_fig, L2_ax = plot_setup("Order", "Normalised $L^2$ Error",
#                                             y_log=True, by=2)
#                 Linf_fig, Linf_ax = plot_setup("Order",
#                                                "Normalised $L^{\\infty}$ Error",
#                                                y_log=True, by=2)
                
#                 for j, flux in enumerate(fluxes):
#                     vals = [error_norms[k+1][j, :] for k in range(len(orders))]
#                     vals = np.array(vals)
#                     L1_ax.plot(orders, vals[:, 0], 'x-', label=flux)
#                     L2_ax.plot(orders, vals[:, 1], 'x-', label=flux)
#                     Linf_ax.plot(orders, vals[:, 2], 'x-', label=flux)
                    
#                 pt.show()
        
        else:
            element_edges = element_edge * (1/2)**np.array([0, 1, 2, 3, 4])
            xticklabels_ = [f'$h_0\\times2^{{-{n}}}$' for n in \
                                range(len(element_edges))]
            xticklabels_[0] = '$h_0$'
            
            for length_ in [.15, .3]:
                folder_name = f'BoundaryValue_Problem/\
order={default_order}/Domain={length_*1e-3*param.L_R:.0f}km/Slope={slope}'
                bbox = (-length_/2, length_/2, 0, length_)
                error_arr = np.zeros((len(element_edges), len(fluxes), 3))
                time_arr = np.zeros((len(element_edges), len(fluxes)))
                for i, h_ in enumerate(element_edges):
                    times, errors = kelvin.kelvin_boundaryvalue_problem(
                        param,
                        bbox,
                        order=default_order,
                        h_target=h_,
                        tolerance=.1,
                        slope_profile=slope,
                        numerical_fluxes=fluxes,
                        theta_value=.5,
                        error_base=2,
                        plot_exact=False,
                        verbose=True
                        )
                    error_arr[i] = errors
                    time_arr[i] = times
                from ppp.Plots import plot_setup, save_plot
                import matplotlib.pyplot as pt
                powers =  np.array(list(range(len(element_edges))))
                fig, ax = plot_setup('Element Edge Length',
                                     'Time (s)',
                                     y_log=True,
                                     scale=2,
                                     by=2)
                ax.plot(powers, time_arr, 'x-')
                
                if plot_scaling:
                    ax.plot(powers, 2.0**(-2 + 4*powers), 'k:')


                ax.set_xticks(powers)
                ax.set_xticklabels(xticklabels_)
                ax.legend(fluxes, fontsize=16)
                pt.show()
                # save_plot(fig, ax, 'Time', folder_name=folder_name,
                #           labels=fluxes)
                
                
                # for i, norm, name, scaling in zip(list(range(3)),
                #                                  ['L1', 'L2', 'Linf'],
                #                                  ['L^1', 'L^2', 'L^{\\infty}'],
                #                                  [2, 2, 1]):
                #     fig, ax = plot_setup('Order', f'${name}$ Error',
                #                          y_log=True, by=2)
                #     ax.plot(powers, error_arr[:, :, i], 'x-')
                    
                #     if plot_scaling:
                #         ax.plot(powers, 2.0**(-(11 + scaling*powers)), 'k:')

                #     ax.set_xticks(powers)
                #     ax.set_xticklabels(xticklabels_)
                #     ax.legend(fluxes, fontsize=16)
                #     pt.show()
                    # save_plot(fig, ax, norm, folder_name=folder_name,
                    #           labels=fluxes)

if __name__ == '__main__':
    from barotropicSWEs.Configuration import configure
    param = configure.main()
    for h in [2.5e-2, 5e-2]: #element edge size
        validation_cases(param, #Default parameter values stored here
                          case="kelvin", #Test cases ['rotating', 'nonrotating', 'kelvin']
                          refinement="p", #Refinement ['h', 'p']
                          default_order=2, #Integer value > 0 used in h-refinement,
                          element_edge=h,
                          slope="cos_squared", # Slope in ['cos_squared', 'gg07']
                          plot_scaling=True)
