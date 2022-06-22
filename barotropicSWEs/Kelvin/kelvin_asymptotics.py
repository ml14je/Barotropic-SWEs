#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Fri Sep 11 15:54:07 2020
"""
import numpy as np

class Kelvin_Asymptotics(object):
    def __init__(self, param, h_func, ηF=1, ω=None, k=None, order=0, λ=None, y=None):
        """
        Parameters
        ----------
        param : TYPE
            DESCRIPTION.
        ηF : float, optional
            Approximate SSH displacement at coastal shelf. The default is 1.

        Returns
        -------
        None.

        """
        self.order = order
        self.param = param
        self.ηF = ηF
        self.k = self.param.L_R*self.param.k if k is None else k
        self.ω = self.param.ω/self.param.f if ω is None else ω

        self.λ = .03 if λ is None else λ
        self.λρ = (self.param.ρ_max-self.param.ρ_min)/self.param.ρ_max
        self.d = self.param.H_pyc
        self.δ = self.param.H_C/self.param.H_D
        self.hC_func = lambda y : h_func(y)/self.param.H_D
        self.Ny = self.param.Ny
        self.L = self.param.Ly/param.L_R
        self.dy = self.L/self.Ny

        if y is None:
            self.y = np.concatenate((np.linspace(0, self.λ, self.Ny),
                                     np.linspace(self.λ, self.L, 200)[1:]))
        else:
            self.y = y

        self.N = len(self.y)
        self.C_inds = self.y <= self.λ
        self.integration_setup()

        self.D_inds = np.invert(self.C_inds)
        self.NC = len(self.C_inds[self.C_inds==True])
        self.Y = self.y[self.C_inds]/self.λ

        self.topography()

    def integration_setup(self, λ=None):
        from ppp.Numerical_Integration import trapezium_matrix
        from scipy.sparse import csr_matrix as sp

        if λ is None:
            λ = self.λ

        M_trap = np.zeros((self.N, self.N))
        M_trap[1:] = trapezium_matrix(self.y).toarray()
        self.M_trap = sp(M_trap)

    def test(self, N=30):
        from ppp.Plots import plot_setup
        from ppp.Newton_Raphson import newton_raphson
        self.hC_func = lambda y : self.δ
        λ_vals = 10**np.linspace(-4, -.5, N)

        ω_vals = np.empty((self.order+2, N))
        ω_analytical, ω_exact = np.zeros((self.order+1, N)), np.zeros(N)
        coeffs = [1, -(1-self.δ), (2-(2+self.k**2)*(1-self.δ))*(1-self.δ)/2]
        for i, c in enumerate(coeffs):
            ω_analytical[i:] += c*(λ_vals**i)
        ω_analytical *= self.k

        def exact_dispersion(λ, ω, k):
            l_D = np.sqrt(k**2-(ω**2-1)+0j)
            l_C= np.sqrt(0j+k**2-(ω**2-1)/self.δ)

            if λ == 0:
                return ω*(ω*l_D-k)/(ω**2-1)

            else:
                return (((ω*l_C+k)*np.exp(-2*l_C*λ)+(ω*l_C-k))*(ω*l_D-k)-self.δ*((ω*l_C+k)*\
                    (ω*l_C-k)*(np.exp(-2*l_C*λ)-1)))/(l_C*(ω-np.sign(k)))


        for i, λ in enumerate(λ_vals):
            self.y = np.concatenate((np.linspace(0, λ, self.Ny), np.linspace(λ, 2*λ, 200)[1:]))
            self.L = self.y[-1]
            self.integration_setup(λ)
            self.topography(λ)
            self.compare()

            #Asymptotic and numerical
            ω_vals[:, i] = np.append(self.ω, self.ω_exact)

            #Exact solutions
            root_func = lambda ω : exact_dispersion(λ, ω, self.k)
            ω_exact[i] = newton_raphson(root_func, self.ω[-1]).real


        fig, ax = plot_setup('$\\lambda$', '$\\omega$',
            scale=self.param.plot_scale, x_log=True)
        lineObjects = ax.plot(λ_vals, ω_vals.T)
        ax.plot(λ_vals, ω_analytical.T, 'k:')
        ax.plot(λ_vals, ω_exact, 'k:')

        import matplotlib.pyplot as pt
        labs = [f'Order {i}' for i in range(self.order+1)] + ['Numerical']
        ax.legend(iter(lineObjects), labs, fontsize=16)
        pt.show()

        fig, ax = plot_setup('$\\epsilon$', 'Relative Error',
                             scale=self.param.plot_scale, x_log=True, y_log=True)
        lineObjects = ax.plot(λ_vals, np.abs((ω_vals[:-1]-ω_vals[-1])).T)
        ax.legend(iter(lineObjects), labs[:-1], fontsize=16)
        pt.show()

    def vary_λ(self, N=30):
        from ppp.Plots import plot_setup


        λ_vals = 10**np.linspace(-4, -.5, N)
        ω_vals = np.empty((self.order+2, N))
        err_vals = np.empty((N, 3))

        for i, λ_ in enumerate(λ_vals):
            self.y = np.concatenate((np.linspace(0, λ_, self.Ny), np.linspace(λ_, 2*λ_, 200)[1:]))
            self.L = self.y[-1]
            self.integration_setup(λ_)
            self.topography(λ_)

            err_vals[i, :] = self.compare(plot=True, λ=λ_, error=True)
            ω_vals[:, i] = np.append(self.ω,  self.ω_exact)

        fig, ax = plot_setup('$\\epsilon$', '$\\omega$',
                             scale=self.param.plot_scale, x_log=True)
        lineObjects = ax.plot(λ_vals, ω_vals.T)

        import matplotlib.pyplot as pt
        labs = [f'Order {i}' for i in range(self.order+1)] + ['Numerical']
        ax.legend(iter(lineObjects), labs, fontsize=16)
        pt.show()

        fig, ax = plot_setup('$\\epsilon$', 'Relative Error',
                             scale=self.param.plot_scale, x_log=True, y_log=True)
        lineObjects = ax.plot(λ_vals, np.abs((ω_vals[:-1]-ω_vals[-1])).T)
        ax.legend(iter(lineObjects), labs[:-1], fontsize=16)
        pt.show()

        fig, ax = plot_setup('$\\epsilon$', 'Infinity Norm',
                             scale=self.param.plot_scale, x_log=True, y_log=True)
        lineObjects = ax.plot(λ_vals, err_vals)
        ax.legend(iter(lineObjects), ['Along-Shore Velocity', 'Cross-Shore Velocity',
                'Surface Displacement'],
                  fontsize=16)
        pt.show()

    def topography(self, λ=None):
        if λ is None:
            λ = self.λ

        C_inds = self.y <= λ

        self.h = np.ones(self.N)
        y_C = self.y[C_inds]

        self.h[C_inds]=self.hC_func(y_C)


    def h_func(self, y, λ=None):
        if λ is None:
            λ = self.λ

        if y < λ:
            return self.hC_func(y)

        else:
            return 1

    def dispersion_relation(self, λ=None, k=None, ω=None):

        if λ is None:
            λ = self.λ

        if ω is not None:
            ω_ = ω
            self.ks = np.zeros(self.order+1)
            self.ks[0] = 1
            self.k1 = +self.integrate(1-self.h)[-1, 0]/λ

            if self.order > 0:
                self.ks[1] = self.k1


            self.k_orders = np.zeros(self.ks.shape)
            for i, k in enumerate(self.ks):
                self.k_orders[i:] += ω_ * self.ks[i] * (λ**i)

            self.k_apprx = self.k_orders[self.order]
            self.ω_apprx = ω_

        else:
            k_ = k if k is not None else self.k
            self.ωs = np.zeros(self.order+1)
            self.ωs[0] = 1
            self.ω1 = -self.integrate(1-self.h)[-1, 0]/λ

            if self.order > 0:
                self.ωs[1] = self.ω1

            if self.order > 1:
                self.ωs[2] = -self.integrate(
                    self.ω1 * (1 + self.h) + 2 * self.integrate(
                        1 - self.h
                        )[:, 0]/λ
                    )[-1, 0]/λ - .5 * (self.ω1 * k_)**2

            self.ω_orders = np.zeros(self.ωs.shape)
            for i, ω in enumerate(self.ωs):
                self.ω_orders[i:] += k_ * self.ωs[i] * (λ**i)

            self.ω_apprx = self.ω_orders[self.order]
            self.k_apprx = k_

    def solutions(self, λ=None):
        if λ is None:
            λ = self.λ

        self.dispersion_relation()
        self.cross_shore(λ)
        self.surface_displacement(λ)
        self.along_shore(λ)
        self.sols_apprx = np.array([self.u, self.v, self.η])

    def cross_shore(self, λ):
        self.Q_outer = np.zeros(self.N, dtype=complex)
        self.Q_inner = np.zeros(self.N, dtype=complex)
        self.Q_match = np.zeros(self.N, dtype=complex)

        if self.order > 0:
            self.Q_outer += -1j * self.k * self.ωs[1] * np.exp(-self.y) * \
                λ
            self.Q_inner += 1j * self.k * self.integrate(1-self.h)[:, 0]
            self.Q_match += -1j * self.k * self.ωs[1] * λ

        if self.order > 1:
            pass

        self.Q = self.Q_inner + self.Q_outer - self.Q_match
        self.v = self.ηF*self.Q/self.h

    def surface_displacement(self, λ):
        self.η_outer = np.exp(-self.y)
        self.η_inner = np.ones(self.N, dtype=complex)
        self.η_match = np.exp(-self.y)

        if self.order > 0:

            self.η_outer += self.ωs[1] * self.k**2 * self.y * \
                            np.exp(-self.y) * λ
            self.η_inner += -self.y
            self.η_match = 1 - self.y

        if self.order > 1:
            pass

        self.η = self.η_inner + self.η_outer - self.η_match
        self.η *= self.ηF

    def along_shore(self, λ):
        # from ppp.Plots import plot_setup
        # import matplotlib.pyplot as pt
        self.u = np.zeros(self.N, dtype=complex)
        self.u += np.exp(-self.y)

        if self.order > 0:
            self.u += (((np.exp(-self.y) - 1) * self.ω1 - \
                    self.integrate(1-self.h)[:, 0]/λ)/self.h + \
                  np.exp(-self.y)*(self.y * self.k**2 - 1) * \
                self.ω1) * λ

        self.u *= self.ηF

    def integrate(self, f_vals):

        if len(f_vals.shape) == 1:
            return self.M_trap @ f_vals[:, None]
        else:
            return self.M_trap @ f_vals

    def exact(self, solutions=True, λ=None, k=None, ω=None):
        from ppp.Shooting import shooting_method2

        if ω is None:
            self.k = self.k if k is None else k
            ω_ = None
            unknown = "WaveFrequency"

        else:
            self.ω = ω
            unknown = "WaveNumber"

        if λ is None:
            λ = self.λ

        def BC(a1, a2):
            return (a1-a2)[:2]


        def vary(X, v1, v2):
            # Varying parameters: X = [ω, η(0)]
            if unknown == "WaveFrequency":
                ω_ = X[0] # wave frequency
                k_ = self.k
                v1[-1], v2[-1] = ω_, ω_ #updating wave frequency

            else:
                k_ = X[0]
                ω_= self.ω
                v1[-1], v2[-1] = k_, k_ #updating wavenumber

            v1[1] = X[1] #Main variable - surface displacement at coastal boundary
            l_D = np.sqrt(1 + k_**2 - ω_**2) #Cross-shore wavenumber in deep ocean
            v2[1] = np.exp(-l_D*self.L) #prescribed surface dispalcement at edge of domain, setting decay condition: Exp(-l_D * L)
            v2[0] = -1j*((k_-l_D*ω_)/(ω_**2-1))*v2[1] #Corresponding v(y=L)

            return v1, v2

                

        def SWEs(y, a):
            aa = np.copy(a) # a = [Qy, η, unknown]
            h = self.h_func(y, λ)

            if unknown == "WaveFrequency":
                ω_ = a[2] # Wave Frequency
                k_ = self.k

            else:
                k_ = a[2] # Wavenumber
                ω_ = self.ω
                

            Qx = (1j*a[0] + k_ * h * a[1])/ω_ # Qx = (i*Qy + k*h(y)*η)/ω
            aa[0] = 1j * (ω_ * a[1] - k_ * Qx) # dQy/dy = i(ω*η - k Qx)
            aa[1] = (1j * ω_ * a[0] - Qx)/h # dη/dy = (i*ω*Qy - Qx)/h(y)
            aa[2] = 0 # dω/dy = 0

            return aa

        if unknown == 'WaveFrequency':
            try:
                vec0 = np.array([self.ω_apprx, 1])

            except AttributeError:
                self.dispersion_relation(k=self.k)
                vec0 = np.array([self.ω_apprx, 1])

            vec1 = np.array([0, 1, self.ω_apprx], dtype=complex)
            vec2 = np.array([0, np.exp(-self.L), self.ω_apprx],
                            dtype=complex)

        else:
            try:
                vec0 = np.array([self.k_apprx, 1])

            except AttributeError:
                self.dispersion_relation(ω=self.ω)
                vec0 = np.array([self.k_apprx, 1])

            vec1 = np.array([0, 1, self.k_apprx], dtype=complex)
            vec2 = np.array([0, np.exp(-self.L), self.k_apprx],
                            dtype=complex)

        shooter = shooting_method2(0, self.L, λ, vec1, vec2, vec0, SWEs, BC, vary,
                                   atol=1e-15, rtol=1e-15)
        exact = shooter.bvp(self.y).real

        if unknown == 'WaveFrequency':
            self.ω_exact = exact
            self.k_exact = self.k

        else:
            self.k_exact = exact
            self.ω_exact = self.ω

        if solutions:
            sols = shooter.solutions.T
            coeff = self.ηF/sols[1, 0]
            self.u_true = coeff*(1j*sols[0] + self.k_exact * self.h * sols[1])/\
                                (self.h * self.ω_exact)
            self.v_true = coeff * sols[0]/self.h
            self.η_true = coeff * sols[1]

            self.sols_true = np.array([self.u_true, self.v_true, self.η_true])

        return self.ω_exact, self.k_exact

    def compare(self, plot=True, error=False, λ=None):
        from ppp.Plots import subplots, set_axis
        import matplotlib.pyplot as pt
        import matplotlib.ticker as mtick

        if λ is None:
            λ = self.λ

        self.solutions(λ)
        self.exact(λ=λ)

        coeff = np.abs(self.η_true[self.Ny])/np.abs(self.η[self.Ny])
        for sols in [self.v, self.η, self.u]:
            sols *= coeff

        if plot:
            fig, axis = subplots(4, 1, scale=self.param.plot_scale, x_share=True)
            for i, var_true, var_apprx, lab in zip(range(3),
                [self.u_true, 1j*self.v_true, self.η_true],
                [self.u, 1j*self.v, self.η],
                ['u', 'v', '\\eta']
                ):
                ax = axis[i]
                set_axis(ax, y_label=f'${lab}(y)$', scale=self.param.plot_scale)
                p1, = ax.plot(self.y, (var_true).real, 'r-')
                p2, = ax.plot(self.y, (var_apprx).real, 'k:')
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

            ax = axis[-1]
            set_axis(ax, x_label='$y$', y_label='$-h(y)$',
                     scale=self.param.plot_scale)
            ax.fill_between(self.y, -self.h, alpha=.1, color='blue')
            for val in [0, λ, self.L]:
                ax.axvline(val, color='k', linestyle='--')

            ax.set_xlim(self.y[[0, -1]])
            ax.legend([p1, p2], ['Numerical', 'Analytical'],
                      loc='lower center', ncol=2, fontsize=16*self.param.plot_scale,
                      bbox_to_anchor=(.5, -1))
            if self.param.save_plot:
                from ppp.Plots import save_plot
                save_plot(fig, axis, f'Slope_eps={λ}_Ly={self.y[-1]:.0f}',
                          'Figures\Composite Solutions')

            else:
                # import matplotlib.pyplot as pt
                pt.show()

        if error:
            err = []
            for vals, vals_true in zip(self.sols_apprx, self.sols_true):
                err.append(np.max(np.abs(vals-vals_true)))

            return err

def step_profile(param, λ=.2):
    def h(y):
        try:
            if y < 1:
                return param.H_C
            else:
                return param.H_D

        except ValueError:
            hvals = param.H_D * np.ones(len(y))

            indC = y < λ
            hvals[indC] = param.H_C

            return hvals

    kelvin = Kelvin_Asymptotics(param, h, 1.5, order=1, λ=λ)
    print(kelvin.compare(plot=True, error=False, λ=λ))

def linear_profile(param, λ=.2):
    def h(y):
        return param.H_C + y*(param.H_D - param.H_C)/λ

    kelvin = Kelvin_Asymptotics(param, h, 1.5, order=2, λ=λ)
    print(kelvin.compare(plot=True, error=False, λ=λ))

def tanh_profile(param, λ=.2):
    λC = λ*param.L_C/(param.L_C+param.L_S)
    λS = λ - λC

    H0, ΔH = (param.H_D+param.H_C)/2, (param.H_D-param.H_C)/2

    def h(y):
        return H0 + ΔH * np.tanh(4*(y-λC-.5*λS)/λS) #Slope profile (Teeluck, 2013)

    kelvin = Kelvin_Asymptotics(param, h, 1.5, order=1, λ=λ)
    print(kelvin.compare(plot=True, error=True, λ=λ))

def GG07_profile(param, λ):
    d = param.H_pyc
    g_p = param.reduced_gravity
    c_inf = np.sqrt(g_p * d)
    cC, cD = c_inf * np.sqrt(1 - d/param.H_C), c_inf * np.sqrt(1 - d/param.H_D)
    λC = λ*param.L_C/(param.L_C+param.L_S)
    λS = λ - λC

    def c1(y): #non-dimensional y
        c_vals = cC + (cD-cC)*(y-λC)/λS

        try:
            c_vals[y<λC] = cC
            c_vals[y > λ] = cD

        except TypeError:
            if y < λC:
                c_vals = cC

            elif y > λ:
                c_vals = cD

        return c_vals

    def h(y): #Non-dimensional y
        return d/(1 - (c1(y)/c_inf)**2)
    y = np.linspace(0, .5, 1001)
    import matplotlib.pyplot as pt
    pt.plot(y, -h(y))
    pt.show()

    kelvin = Kelvin_Asymptotics(param, h, 1.5, order=2, λ=λ)
    kelvin.solutions(λ=λ)
    # print(kelvin.compare(plot=True, error=True, λ=λ))


    # import matplotlib.pyplot as pt

    v = (1j*kelvin.v).real
    pt.plot(kelvin.y, v*kelvin.h, 'y-') #Volume Flux
    N = 500

    ys = np.linspace(0, λC+λS, N+1)
    from ppp.Numerical_Integration import trapezium_matrix
    M_int = np.zeros((N+1, N+1))
    M_int[1:] = trapezium_matrix(ys).toarray()
    print(λ, h(0), h(λC), h(λC+λS))

    #Approximate Kelvin Volume Flux
    Q1 = .4 * λ *(kelvin.ωs[1]*(np.exp(-ys) - 1) - M_int @ (1-kelvin.hC_func(ys))/λ)
    Q2 = .4 * λ *kelvin.ωs[1] * np.ones(len(ys))
    # #Basin topography
    # pt.plot(kelvin.y, kelvin.h, 'r')
    # #Coastal Topography
    # pt.plot(ys, kelvin.hC_func(ys), 'b')

    for y_val in [0, λC, λ]:
        pt.axvline(y_val, color='k', linestyle=':')
    pt.show()
    Q3 = .4 * kelvin.ωs[1]* ys * (1 - λ)
    Q4 = .4 * λ * kelvin.ωs[1] * ys/λ
    #.4 * λ * kelvin.ωs[1] * ys * (1 - λ)/λ
    # Q4 = 2* λ *(kelvin.ωs[1]*(-ys) - ys/λ + M_int @ (kelvin.hC_func(ys))/λ)

    #Volume Fluxes
    from ppp.Plots import plot_setup
    print(kelvin.h, v)
    fig, ax = plot_setup('$y$', 'Volume Flux [$\\rm{m^2/s}$]')

    ax.plot(kelvin.y, .2 * param.H_D * v*kelvin.h, linewidth=3,
            label='Numerical') #Volume Flux
    ax.plot(ys, param.H_D * Q1, linewidth=3,
            label='Analytical') #Volume Flux
    ax.plot(ys, param.H_D * Q3, linewidth=3,
            label='Linear Approximation')
    ax.plot(ys, param.H_D * Q4, linewidth=3,
            label='Craig (1987) Model')

    for y_val in [0, λC, λ]:
        ax.axvline(y_val, color='k', linestyle=':')

    ax.set_xticks([0, λC, λ])
    ax.set_xticklabels(['0', '$L_C$', '$\\lambda=L_C+L_S$'])
    ax.legend(fontsize=16, loc=1)

    ax.set_xlim([0, λ])
    pt.show()

    # # int1 = 1 - ys2/λ - (param.H_C/param.H_D) * \
    # #         (param.H_C/param.H_D) * (
    # #             (np.log(1 + (A1 * λ + A2)) - np.log(1 - (A1 * λ + A2)))
    # #         -   (np.log(1 + (A1 * ys2 + A2)) - np.log(1 - (A1 * ys2 + A2)))
    # #         )/(2 * A2 * λ)
    # # int2 = M_int2 @ (1-kelvin.hC_func(ys2[:, None]))[:, 0] / λ

    # pt.plot(ys2, int1)
    # pt.plot(ys2, int2)
    # pt.show()


if __name__ == '__main__':
    import config_param
    param = config_param.configure()
    param.Ly = 2*param.L_R
    param.Ny = 1000 #Number of grid points along coastal domain
    param.H_D = 4000 # h(y) as y -> \infty
    param.H_C = 200 # h(y=0)
    param.r = 0 #Zero Rayleigh-friction
    param.save_plot=False

    λ = 2.8e-2
    from real_topography import data_model
    # lat, lon = 39, -125
    # y_vals, h_true = data_model(lat, lon, param)
    # y_vals = np.linspace(0, .2, 101)

    # import matplotlib.pyplot as pt
    # pt.plot(y_vals, h_true(y_vals*param.L_R))
    # pt.show()

    # raise ValueError


    # h_func = lambda y: h_true(y*param.L_R)
    # kelvin = Kelvin_Asymptotics(param, h_func, λ=λ, order=2)
    # kelvin.solutions(λ=λ)
    # print(kelvin.compare(plot=True, error=True, λ=λ))

    # GG07_profile(param, λ)
    # baroclinic_GG()

    # for case in [GG07_profile]: # step_profile, linear_profile, tanh_profile,
    #     case(param, λ)

    