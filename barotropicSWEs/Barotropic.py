#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Tue Mar  9 16:00:06 2021
"""
import numpy as np
import plotly.io as pio

pio.renderers.default = "browser"


class solver(object):
    def __init__(
        self,
        fem,
        param=None,
        h_func=None,
        periods=2,
        flux_scheme="central",
        boundary_conditions="Solid Wall",
        rotation=True,
        θ=1,
        background_flow=None,
        wave_frequency=1.4,
        rayleigh_friction=0.05,
    ):
        from scipy.sparse import diags

        self.rotation = rotation
        self.fem = fem
        self.param = param
        self.ω = wave_frequency
        self.r = rayleigh_friction

        self.X, self.Y = self.fem.x.flatten("F"), self.fem.y.flatten("F")

        # Normal matrices:
        self.Nx = diags(self.fem.nx.T.flatten(), format="csr")
        self.Ny = diags(self.fem.ny.T.flatten(), format="csr")
        self.scheme, self.boundary_conditions = (
            flux_scheme.upper(),
            boundary_conditions.upper(),
        )

        if self.scheme not in [
            "CENTRAL",
            "UPWIND",
            "LAX-FRIEDRICHS",
            "RIEMANN",
            "PENALTY",
            "ALTERNATING",
        ]:
            raise ValueError("Invalid Flux scheme")

        if self.boundary_conditions not in [
            "SOLID WALL",
            "OPEN FLOW",
            "MOVING WALL",
            "SPECIFIED",
        ]:
            raise ValueError("Invalid Boundary Condition")

        else:
            # Schemes as given in Hesthaven & Warburton
            if self.scheme == "CENTRAL":
                self.α, self.β, self.γ, self.θ = 0, 0, 0, 0.5

            elif self.scheme == "UPWIND":
                self.α, self.β, self.γ, self.θ = 0, 1, 1, 0.5

            elif self.scheme == "PENALTY":
                self.α, self.β, self.γ, self.θ = 1, 0, 1, 0.5

            elif self.scheme == "LAX-FRIEDRICHS":
                self.α, self.β, self.γ, self.θ = 1, 1, 1, 0.5

            else:  # Alternating flux as given in (Shu, 2002) -certainly (Ambati & Bokhove, 2007), or Riemann Flux
                self.α, self.β, self.γ, self.θ = 0, 0, 0, θ

        if h_func is None:
            self.h_func = np.vectorize(lambda X, Y: 1)
        else:
            self.h_func = h_func

        self.h = self.h_func(self.X, self.Y)

        if background_flow is None:
            self.background_flow = np.zeros((len(self.X) * 2, 1))

        else:
            self.background_flow = np.concatenate(
                (background_flow[0][:, None], background_flow[1][:, None]), axis=0
            )

        self.matrix_setup()

    def initial_value_problem(
        self,
        initial_values,
        periods=10,
        Nout=None,
        file_name=None,
        method="RK4",
        ω=None,
        φ=0,
        animate=False,
        N_frames=0,
    ):
        from ppp.Jacobi import JacobiGQ
        from numpy.linalg import norm

        # Compute time-step size
        rLGL = JacobiGQ(0, 0, self.fem.N)[0]
        rmin = norm(rLGL[0] - rLGL[1])
        dtscale = self.fem.dtscale
        dt = 0.5 * np.min(dtscale) * rmin * 2 / 3

        from math import ceil

        if Nout is None:
            Nout = self.fem.N

        if file_name is None:
            file_name = f"Solutions: {self.scheme}"

        ω = self.ω if ω is None else ω

        # Period of potential forcing φ of frequency ω
        T = 2 * np.pi / ω

        t_final = periods * T
        N = self.fem.Np * self.fem.K
        from scipy.sparse import csr_matrix as sp
        from scipy.sparse import identity, block_diag

        i, o = identity(N), sp((N, N))
        I2 = block_diag(2 * [i] + [o])
        assert ω is not None

        if φ == 0:
            φ = np.zeros(len(self.X))

        if np.all(initial_values) == 0:
            F = (
                lambda t: self.forcing(φ)
                * np.exp(-1j * ω * t)
                * np.tanh(t / (4 * T)) ** 4
            )
            r_ = lambda t: self.r * (1 - np.tanh(t / (4 * T)) ** 4)

        else:
            F = lambda t: self.forcing(φ) * np.exp(-1j * ω * t)
            r_ = lambda t: self.r

        def rhs(t, y):
            return (self.A - r_(t) * I2) @ y + F(t)

        if method in [
            "Forward Euler",
            "Explicit Midpoint",
            "Heun",
            "Ralston",
            "RK3",
            "Heun3",
            "Ralston3",
            "SSPRK3",
            "RK4",
            "3/8 Rule",
            "RK5",
        ]:
            from ppp.Explicit import explicit

            timestepping = explicit(
                rhs,
                initial_values,
                0,
                t_final,
                N=ceil(t_final / dt),
                method=method,
                nt=ceil(((t_final / dt) + 1) / (N_frames + 1)),
                verbose=False,
            )

        elif method in [
            "Heun_Euler",
            "Runge–Kutta–Fehlberg",
            "Bogacki–Shampine",
            "Fehlberg",
            "Cash–Karp",
            "Dormand–Prince",
        ]:
            from ppp.Embedded import embedded

            timestepping = embedded(rhs, initial_values, 0, t_final, method=method)

        else:
            raise ValueError("method argument is not defined.")

        self.time = timestepping.t_vals
        sols = timestepping.y_vals[:, :, 0]

        return timestepping.t_vals, sols

    def eigenvalue_problem(self):
        from scipy.linalg import eig

        eig_vals, eig_modes = eig(
            self.A.todense(),
            # b=-1j*np.eye(self.A.shape[0]),
            overwrite_a=False,
            overwrite_b=False,
            check_finite=False,
        )

        return 1j * eig_vals, eig_modes

    def matrix_setup(self):
        from scipy.sparse import bmat as bsp
        from scipy.sparse import csr_matrix as sp
        from scipy.sparse import identity, diags, block_diag

        vmapM, vmapP = self.fem.vmapM, self.fem.vmapP
        # mapM, mapP = self.fem.mapM, self.fem.mapP
        # print(vmapM - vmapP)

        mapB, vmapB = self.fem.mapB, self.fem.vmapB
        N = vmapM.shape[0]
        # print(self.Nfp, self.Nfaces, self.K)
        # print(self.fem.Nfaces * self.fem.K * self.fem.Nfp, N, self.fem.Np*self.fem.K)
        inds = np.arange(N)
        self.Im = sp((N, self.fem.Np * self.fem.K)).tolil()
        self.Ip = sp((N, self.fem.Np * self.fem.K)).tolil()
        # Hm, Hp = Im.copy(), Ip.copy()
        self.Im[inds, vmapM], self.Ip[inds, vmapP] = 1, 1

        # print(self.Im.shape, self.X.shape, np.unique(self.X).shape)
        Im, Ip = self.Im.tocsr(), self.Ip.tocsr()
        self.avg = 0.5 * (self.Im + self.Ip)
        self.H = diags(self.h)

        # Normal matrices:
        Nx, Ny = self.Nx, self.Ny

        N00, N10 = self.fem.nx.shape
        I, ON = identity(N00 * N10), sp((N00 * N10, N00 * N10))

        self.fscale = diags(self.fem.Fscale.T.flatten())
        self.fscale_inv = diags(1 / self.fem.Fscale.T.flatten())
        self.Fscale = block_diag([self.fscale] * 3)
        self.lift = block_diag([self.fem.LIFT] * self.fem.K)
        self.LIFT = block_diag([self.lift] * 3)

        self.Dx = diags(self.fem.rx.T.flatten()) @ block_diag(
            [self.fem.Dr] * N10
        ) + diags(self.fem.sx.T.flatten()) @ block_diag([self.fem.Ds] * N10)

        self.Dy = diags(self.fem.ry.T.flatten()) @ block_diag(
            [self.fem.Dr] * N10
        ) + diags(self.fem.sy.T.flatten()) @ block_diag([self.fem.Ds] * N10)

        ON = sp((Ny.shape[0], Ip.shape[1]))
        i = identity(self.fem.Np * self.fem.K)
        o = sp((self.fem.Np * self.fem.K, self.fem.Np * self.fem.K))

        # System Equations:
        if self.rotation:
            self.A1 = bsp(
                [
                    [o, i, -self.Dx],
                    [-i, o, -self.Dy],
                    [-self.Dx @ self.H, -self.Dy @ self.H, o],
                ]
            )

        else:
            self.A1 = bsp(
                [
                    [o, o, -self.Dx],
                    [o, o, -self.Dy],
                    [-self.Dx @ self.H, -self.Dy @ self.H, o],
                ]
            )

        # Boundary Conditions:
        Ipu1, Ipu2, Ipu3 = Ip.copy().tolil(), ON.copy().tolil(), ON.copy().tolil()
        Ipv1, Ipv2, Ipv3 = ON.copy().tolil(), Ip.copy().tolil(), ON.copy().tolil()
        Ipη1, Ipη2, Ipη3 = ON.copy().tolil(), ON.copy().tolil(), Ip.copy().tolil()

        N = self.fem.K * self.fem.Nfaces
        self.Un = np.zeros((3 * Nx.shape[0], 1))

        if self.boundary_conditions == "SOLID WALL":
            #            if self.rotation and self.scheme not in ['ALTERNATING', 'CENTRAL']:
            # η+ = η-
            # Ipη3[mapB, vmapB] = Im[mapB, vmapB]

            # u+ = (ny^2 - nx^2) * -u- - 2 * nx * ny * v- (non-dimensional impermeability)
            Ipu1[mapB, vmapB] = ((Ny @ Ny - Nx @ Nx) @ Im)[mapB, vmapB]
            Ipu2[mapB, vmapB] = -2 * (Nx @ Ny @ Im)[mapB, vmapB]

            # v+ = (nx^2 - ny^2) * -v- - 2 * nx * ny * u- (non-dimensional impermeability)
            Ipv2[mapB, vmapB] = ((Nx @ Nx - Ny @ Ny) @ Im)[mapB, vmapB]
            Ipv1[mapB, vmapB] = -2 * (Nx @ Ny @ Im)[mapB, vmapB]

        #          else:
        # Case of Ambati & Bokhove to constrain central flux on domain boundary
        #              raise ValueError
        #              # η+ = η-
        #              Ipη3[mapB, vmapB] = Im[mapB, vmapB]

        elif self.boundary_conditions == "OPEN FLOW":
            # [[η]]=0 ==> η+ = η-
            Ipη3[mapB, vmapB] = Im[mapB, vmapB]

        elif self.boundary_conditions == "FIXED FLUX":
            # {{h u}} = Q(x) ==> u+ = -u- +2Q(x)/h(x)
            Ipu1[mapB, vmapB] = -Im + 2 * self.Q(self.X, self.Y) / self.h(
                self.X, self.Y
            )

        elif self.boundary_conditions == "SPECIFIED":
            for bc in self.fem.BCs.keys():
                m, vm = self.fem.maps[bc], self.fem.vmaps[bc]
                if "Wall" or "Open" in bc:
                    # η+ = η-
                    # Ipη3[m, vm] = Im[m, vm]

                    # u+ = (ny^2 - nx^2) * -u- - 2 * nx * ny * v- (non-dimensional impermeability)
                    Ipu1[m, vm] = ((Ny @ Ny - Nx @ Nx) @ Im)[m, vm]
                    Ipu2[m, vm] = -2 * (Nx @ Ny @ Im)[m, vm]

                    # v+ = (nx^2 - ny^2) * -v- - 2 * nx * ny * u- (non-dimensional impermeability)
                    Ipv2[m, vm] = ((Nx @ Nx - Ny @ Ny) @ Im)[m, vm]
                    Ipv1[m, vm] = -2 * (Nx @ Ny @ Im)[m, vm]

                if "Open" in bc:
                    N = self.fem.vmapM.shape[0]
                    Norms = bsp([[Nx, Ny]])
                    IO = sp((N, self.fem.Np * self.fem.K)).tolil()
                    IO[m, vm] = 1
                    IO = block_diag([IO] * 2)
                    U = Norms @ IO @ block_diag([self.H] * 2) @ self.background_flow
                    Qx = 2 * Nx @ U
                    Qy = 2 * Ny @ U
                    η_t = -1j * self.ω * sp(U.shape)  # zero
                    X = bsp([[Qx], [Qy], [η_t]])
                    self.Un += X

        else:  # 'MOVING WALL'
            # η+ = (-nt^2 + nx^2 + ny^2)η- - 2 * nt * nx * h- u- - 2 * nt * ny * h- v-
            # u+ = (-nt^2 + nx^2 + ny^2)η- - 2 * nt * nx * h- u- - 2 * nt * ny * h- v-
            # v+ = (-nt^2 + nx^2 + ny^2)η- - 2 * nt * nx * h- u- - 2 * nt * ny * h- v-
            pass

        Ipu = bsp([[Ipu1, Ipu2, Ipu3]])
        Ipv = bsp([[Ipv1, Ipv2, Ipv3]])
        Ipη = bsp([[Ipη1, Ipη2, Ipη3]])
        Ip2 = bsp([[Ipu], [Ipv], [Ipη]])
        Im2 = block_diag(3 * [Im])

        self.jump = Im2 - Ip2  # Jump operator
        H_mat = block_diag(
            [self.H, self.H, i]
        )  # allows volume transport flux in mass conservation

        α, β, γ, θ = self.α, self.β, self.γ, self.θ
        θ = θ * np.ones(Ny.shape[0])
        θ[mapB] = 0.5

        h_av = (self.h[vmapP] + self.h[vmapM]) / 2

        c = diags(np.sqrt(h_av))
        H_inv = diags(1 / h_av)

        if self.scheme == "RIEMANN":
            raise ValueError("Riemann Flux not yet implemented")

        else:
            self.Flux = 0.5 * bsp(
                [
                    [
                        -c @ H_inv @ sp((α * (Nx @ Nx) + β * (Ny @ Ny))),
                        -c @ H_inv @ sp(((α - β) * Nx @ Ny)),
                        2 * diags(1 - θ) @ Nx,
                    ],
                    [
                        -c @ H_inv @ sp(((α - β) * Nx @ Ny)),
                        -c @ H_inv @ sp((β * Nx @ Nx + α * Ny @ Ny)),
                        2 * diags(1 - θ) @ Ny,
                    ],
                    [2 * diags(θ) @ Nx, 2 * diags(θ) @ Ny, -γ * c @ I],
                ]
            )

        # Flux Contributiuons
        self.F = self.LIFT @ self.Fscale @ self.Flux @ self.jump @ H_mat
        self.A = self.A1 + self.F

        # Singular Mass Matrix
        self.M = block_diag([self.fem.mass_matrix] * N10)
        ones = np.ones((self.fem.Np * self.fem.K, 1))

        self.norm = (ones.T @ self.M @ ones)[0, 0]

        # Non-Dimensional energy mass matrix
        self.M_E = block_diag([self.M @ self.H, self.M @ self.H, self.M])

        # Inhomogeneous part from BCs
        self.U_D = self.LIFT @ self.Fscale @ self.Flux @ self.Un

        assert self.A.shape[0] == self.U_D.shape[0]

    def rhs3(self, sols, t):

        return self.A @ sols

    def animate(self, TRI, xout, yout, uout, name="solutions", open_animation=False):
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"is_3d": True}, {"is_3d": True}, {"is_3d": True}]],
            subplot_titles=["$u$", "$v$", "$\\eta$"],
        )

        x, y = xout[None, :], yout[None, :]

        import plotly.graph_objects as go

        scene_dics = []
        for i in range(3):
            z = uout[:, i]
            mesh, lines = plotly_triangular_mesh(
                np.concatenate([x, y, z.real[None, 0]]),
                TRI,
                showscale=True,
                cvals=[-np.max(np.abs(z)), np.max(np.abs(z))],
                x_loc=(-0.05 + (i + 1)) / 3,
                colorscale="RdBu_r",
            )
            fig.add_traces([mesh, lines], rows=[1, 1], cols=[i + 1, i + 1])

            scene_dics.append(
                dict(
                    xaxis=dict(range=[np.min(x), np.max(x)], autorange=False),
                    yaxis=dict(range=[np.min(y), np.max(y)], autorange=False),
                    zaxis=dict(
                        range=[-np.max(np.abs(z)), np.max(np.abs(z))], autorange=False
                    ),
                    aspectmode="cube",
                )
            )

        frames = []
        for i, t in enumerate(self.time):
            datum = []
            for j in range(3):
                z = uout[:, j]
                verts = np.concatenate([x, y, z.real[None, i]])
                mesh, lines = plotly_triangular_mesh(
                    verts,
                    TRI,
                    showscale=True,
                    cvals=[-np.max(np.abs(z)), np.max(np.abs(z))],
                    x_loc=-0.05 + (j + 1) * 0.36,
                    colorscale="RdBu_r",
                )
                datum += [mesh, lines]
            frames.append(
                go.Frame(data=datum, traces=[0, 1, 2, 3, 4, 5], name=f"Frame {i}")
            )

        axis = dict(
            showline=True, zeroline=False, ticklen=4, mirror=True, showgrid=False
        )
        fig.update(frames=frames)

        def get_sliders(n_frames, fr_duration=50, y_pos=-0.1, slider_len=0.9):
            x_pos = (1 - slider_len) / 2
            # n_frames= number of frames
            # fr_duration=duration in milliseconds of each frame
            # x_pos x-coordinate where the slider starts
            # slider_len is a number in (0,1] giving the slider length as a fraction of x-axis length
            return [
                dict(
                    steps=[
                        dict(
                            method="animate",  # Sets the Plotly method to be called when the
                            # slider value is changed.
                            args=[
                                [
                                    f"Frame {k}"
                                ],  # Sets the arguments values to be passed to
                                # the Plotly method set in method on slide
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=fr_duration, redraw=True),
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=f"{(t/self.time[-1]):.2f}",
                        )
                        for k, t in enumerate(self.time)
                    ],
                    transition=dict(duration=0),
                    x=x_pos,
                    y=y_pos,
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix="Time: ",
                        visible=True,
                        xanchor="center",
                    ),
                    len=slider_len,
                )
            ]

        layout = dict(
            title=dict(
                text="Linearised Shallow-Water Equations",
                x=0.5,
                y=0.95,
                font=dict(family="Balto", size=20),
            ),
            showlegend=False,
            autosize=False,
            width=0.8 * 1800,
            height=0.8 * 900,
            xaxis=dict(axis, **{"range": [np.min(x), np.max(x)]}),
            yaxis=dict(axis, **{"range": [np.min(y), np.max(y)]}),
            plot_bgcolor="#c1e3ff",
            hovermode="closest",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )
        layout.update(
            scene=scene_dics[0],
            scene2=scene_dics[1],
            scene3=scene_dics[2],
            coloraxis=dict(
                colorscale="RdBu_r",
                colorbar_x=0.0,
                colorbar_thickness=20,
                cmin=-np.max(np.abs(uout[:, 0])),
                cmax=np.max(np.abs(uout[:, 0])),
                showscale=True,
            ),
            coloraxis2=dict(
                colorscale="RdBu_r",
                colorbar_x=0.33,
                colorbar_thickness=20,
                cmin=-np.max(np.abs(uout[:, 1])),
                cmax=np.max(np.abs(uout[:, 1])),
                showscale=True,
            ),
            coloraxis3=dict(
                colorscale="RdBu_r",
                colorbar_x=0.66,
                colorbar_thickness=20,
                cmin=-np.max(np.abs(uout[:, 2])),
                cmax=np.max(np.abs(uout[:, 2])),
                showscale=True,
            ),
            sliders=get_sliders(len(frames), fr_duration=100, slider_len=0.75),
        )
        fig.update_layout(layout)
        fig.write_html(name + ".html", auto_open=open_animation, include_mathjax="cdn")

    def animate_scalar(
        self, TRI, xout, yout, uout, name="solutions", open_animation=False
    ):
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"is_3d": True}]],
            subplot_titles=["$\\partial_xv-\\partial_yu$"],
        )

        x, y = xout[None, :], yout[None, :]

        import plotly.graph_objects as go

        scene_dics = []

        # Initial Plot
        z = uout
        mesh, lines = plotly_triangular_mesh(
            np.concatenate([x, y, z.real[None, 0]]),
            TRI,
            showscale=True,
            cvals=[-np.max(np.abs(z)), np.max(np.abs(z))],
            x_loc=(-0.05 + 1),
            colorscale="RdBu_r",
        )
        fig.add_traces([mesh, lines], rows=[1, 1], cols=[1, 1])

        scene_dics.append(
            dict(
                xaxis=dict(range=[np.min(x), np.max(x)], autorange=False),
                yaxis=dict(range=[np.min(y), np.max(y)], autorange=False),
                zaxis=dict(
                    range=[-np.max(np.abs(z)), np.max(np.abs(z))], autorange=False
                ),
                aspectmode="cube",
            )
        )

        frames = []
        #        print(z.shape, self.time.shape)
        for i, t in enumerate(self.time):
            datum = []
            z = uout
            verts = np.concatenate([x, y, z.real[None, i]])
            mesh, lines = plotly_triangular_mesh(
                verts,
                TRI,
                showscale=True,
                cvals=[-np.max(np.abs(z)), np.max(np.abs(z))],
                x_loc=-0.05 + 1,
                colorscale="RdBu_r",
            )
            datum += [mesh, lines]
            frames.append(go.Frame(data=datum, traces=[0, 1], name=f"Frame {i}"))

        axis = dict(
            showline=True, zeroline=False, ticklen=4, mirror=True, showgrid=False
        )
        fig.update(frames=frames)

        def get_sliders(n_frames, fr_duration=50, y_pos=-0.1, slider_len=0.9):
            x_pos = (1 - slider_len) / 2
            # n_frames= number of frames
            # fr_duration=duration in milliseconds of each frame
            # x_pos x-coordinate where the slider starts
            # slider_len is a number in (0,1] giving the slider length as a fraction of x-axis length
            return [
                dict(
                    steps=[
                        dict(
                            method="animate",  # Sets the Plotly method to be called when the
                            # slider value is changed.
                            args=[
                                [
                                    f"Frame {k}"
                                ],  # Sets the arguments values to be passed to
                                # the Plotly method set in method on slide
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=fr_duration, redraw=True),
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=f"{(t/self.time[-1]):.2f}",
                        )
                        for k, t in enumerate(self.time)
                    ],
                    transition=dict(duration=0),
                    x=x_pos,
                    y=y_pos,
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix="Time: ",
                        visible=True,
                        xanchor="center",
                    ),
                    len=slider_len,
                )
            ]

        layout = dict(
            title=dict(
                text="Linearised Shallow-Water Equations",
                x=0.5,
                y=0.95,
                font=dict(family="Balto", size=20),
            ),
            showlegend=False,
            autosize=False,
            width=0.8 * 1800,
            height=0.8 * 900,
            xaxis=dict(axis, **{"range": [np.min(x), np.max(x)]}),
            yaxis=dict(axis, **{"range": [np.min(y), np.max(y)]}),
            plot_bgcolor="#c1e3ff",
            hovermode="closest",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )
        layout.update(
            scene=scene_dics[0],
            coloraxis=dict(
                colorscale="RdBu_r",
                colorbar_x=0.0,
                colorbar_thickness=20,
                cmin=-np.max(np.abs(uout[:, 0])),
                cmax=np.max(np.abs(uout[:, 0])),
                showscale=True,
            ),
            sliders=get_sliders(len(frames), fr_duration=100, slider_len=0.75),
        )
        fig.update_layout(layout)
        fig.write_html(name + ".html", auto_open=open_animation, include_mathjax="cdn")

    def evp(self, k_vals=10):
        from scipy.linalg import eig

        vals, vecs = eig(1j * self.A.todense())
        return vals, vecs.T

    def forcing(self, φ):
        from scipy.sparse import block_diag  # , diags
        from scipy.sparse import bmat as bsp
        from scipy.sparse import csr_matrix as sp

        self.φ = bsp([[sp(φ)] * 3]).T

        N = self.fem.Np * self.fem.K
        o = sp((N, N))
        grad = block_diag([self.Dx, self.Dy, o])

        # Potential forcing + prescription of boundary conditions
        return grad @ self.φ - self.U_D

    def boundary_value_problem(
        self,
        φ,
        rayleigh_friction=None,
        wave_frequency=None,
        file_name="BVP Animation",
        animate=True,
        frames=20,
        verbose=True,
    ):
        from scipy.sparse import identity, block_diag  # , diags
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import csr_matrix as sp

        ω = self.ω if wave_frequency is None else wave_frequency
        r = self.r if rayleigh_friction is None else rayleigh_friction
        self.frames = frames
        self.time = np.linspace(0, 2 * np.pi / ω, self.frames + 1)

        N = self.fem.Np * self.fem.K
        i, o = identity(N), sp((N, N))

        I, I2 = block_diag([i] * 3), block_diag(2 * [i] + [o])
        assert ω is not None

        A = -self.A - 1j * ω * I + r * I2
        if verbose:
            print("Solving BVP using spsolve")

        sols = spsolve(A, self.forcing(φ))[None, :]

        if animate:
            if verbose:
                print("Animating and saving BVP")

            sols_t = sols * np.exp(-1j * ω * self.time[:, None])
            self.animate_sols(np.copy(sols_t), file_name=file_name)

        return sols

    def animate_velocity(
        self,
        vec,
        Nout=None,
        name="Velocity_Quiver",
        open_animation=True,
        scale=1,
        arrow_scale=0.1,
        cmap="RdBu_r",
        fr_dur=150,
    ):
        if Nout is None:
            Nout = self.fem.N
        h_max = 0.025

        # scale /= self.fem.N
        arrow_scale *= self.fem.N
        u, v, η = np.split(vec, 3, axis=1)
        u *= self.param.c
        v *= self.param.c
        η *= self.param.H_D
        scale = (h_max / self.fem.N) / np.max(np.sqrt(np.abs(v) ** 2 + np.abs(u) ** 2))

        # project velocity onto linear basis functions
        TRI, xout, yout, u, interp = self.fem.FormatData2D(Nout, self.X, self.Y, u)
        v = (interp @ v.T).T

        TRI2, xout2, yout2, η, interp = self.fem.FormatData2D(Nout, self.X, self.Y, η)

        η_mag = np.max(np.abs(η))

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"is_3d": False}, {"is_3d": True}]],
            subplot_titles=["Velocity", "$\\eta$"],
        )

        x, y = xout[None, :], yout[None, :]
        x2, y2 = xout2[None, :], yout2[None, :]

        from scipy.interpolate import griddata

        xr, yr = np.linspace(np.min(self.X), np.max(self.X), 101), np.linspace(
            np.min(self.Y), np.max(self.Y), 101
        )
        xr, yr = np.meshgrid(xr, yr)
        Z = griddata((self.X, self.Y), self.h, (xr, yr), method="cubic")

        import plotly.graph_objects as go
        import plotly.figure_factory as ff

        scene_dics = []

        quiv = ff.create_quiver(
            x,
            y,
            u.real[None, 0],
            v.real[None, 0],
            scale=scale,
            arrow_scale=arrow_scale,
            line_width=0.25,
        )
        topography = go.Heatmap(
            x=xr[0], y=yr[:, 0], z=Z, colorscale="Blues", visible=True
        )

        mesh, lines = plotly_triangular_mesh(
            np.concatenate([x2, y2, η.real[None, 0]]),
            TRI2,
            showscale=True,
            cvals=[-η_mag, η_mag],
            x_loc=0.95,
            colorscale="RdBu_r",
        )

        fig.add_traces([topography, quiv.data[0]], rows=[1, 1], cols=[1, 1])
        fig.add_traces([mesh, lines], rows=[1, 1], cols=[2, 2])

        frames = []
        for i, t in enumerate(self.time):
            print(i)
            u2, v2 = u.real[i, None, :], v.real[i, None, :]
            mesh, lines = plotly_triangular_mesh(
                np.concatenate([x2, y2, (η[None, i]).real]),
                TRI2,
                showscale=True,
                cvals=[-η_mag, η_mag],
                x_loc=0.95,
                colorscale="RdBu_r",
            )

            quiv = ff.create_quiver(
                x, y, u2, v2, line_width=0.25, scale=scale, arrow_scale=arrow_scale
            )
            datum = [quiv.data[0], mesh, lines]
            frames.append(go.Frame(data=datum, traces=[1, 2, 3], name=f"Frame {i}"))

        axis = dict(
            showline=True, zeroline=False, ticklen=4, mirror=True, showgrid=False
        )
        fig.update(frames=frames)

        def get_sliders(n_frames, fr_duration=fr_dur, y_pos=-0.1, slider_len=0.9):
            x_pos = (1 - slider_len) / 2
            # n_frames= number of frames
            # fr_duration=duration in milliseconds of each frame
            # x_pos x-coordinate where the slider starts
            # slider_len is a number in (0,1] giving the slider length as a fraction of x-axis length
            return [
                dict(
                    steps=[
                        dict(
                            method="animate",  # Sets the Plotly method to be called when the
                            # slider value is changed.
                            args=[
                                [
                                    f"Frame {k}"
                                ],  # Sets the arguments values to be passed to
                                # the Plotly method set in method on slide
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=fr_duration, redraw=True),
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=f"{(t/self.time[-1]):.2f}",
                        )
                        for k, t in enumerate(self.time)
                    ],
                    transition=dict(duration=0),
                    x=x_pos,
                    y=y_pos,
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix="Time: ",
                        visible=True,
                        xanchor="center",
                    ),
                    len=slider_len,
                )
            ]

        layout = dict(
            title=dict(
                text="Linearised Shallow-Water Equations",
                x=0.5,
                y=0.95,
                font=dict(family="Balto", size=20),
            ),
            showlegend=False,
            autosize=False,
            width=0.8 * 1800,
            height=0.8 * 900,
            xaxis=dict(axis, **{"range": [np.min(x), np.max(x)]}),
            yaxis=dict(axis, **{"range": [np.min(y), np.max(y)]}),
            hovermode="closest",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": fr_dur, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )
        scene_dics.append(
            dict(
                xaxis=dict(range=[np.min(x), np.max(x)], autorange=False),
                yaxis=dict(range=[np.min(y), np.max(y)], autorange=False),
                zaxis=dict(range=[-η_mag, η_mag]),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
            )
        )

        layout.update(
            scene=scene_dics[0],
            coloraxis2=dict(
                colorscale=cmap,
                colorbar_x=1,
                colorbar_thickness=20,
                cmin=-η_mag,
                cmax=η_mag,
                showscale=True,
            ),
            sliders=get_sliders(len(frames), fr_duration=fr_dur, slider_len=0.75),
        )
        fig.update_layout(layout)
        fig.write_html(name + ".html", auto_open=open_animation, include_mathjax="cdn")

    def animate_sols(self, vec, Nout=None, file_name="Solution", open_animation=True):
        if Nout is None:
            Nout = self.fem.N
        u, v, η = np.split(vec, 3, axis=1)

        u *= self.param.c
        v *= self.param.c
        η *= self.param.H_D

        TRI, xout, yout, u, interp = self.fem.FormatData2D(Nout, self.X, self.Y, u)
        v, η = (interp @ v.T).T, (interp @ η.T).T
        sols = np.concatenate([u[:, None, :], v[:, None, :], η[:, None, :]], axis=1)

        self.animate(
            TRI, xout, yout, sols, name=file_name, open_animation=open_animation
        )

    def plot_topography(self, Nout=None, plot_type="Contour"):
        assert plot_type.upper() in ["MESH3D", "CONTOUR"]

        if Nout is None:
            Nout = self.fem.N
        TRI, xout, yout, h, interp = self.fem.FormatData2D(
            Nout, self.X, self.Y, self.h[None, :]
        )

        x, y, h = xout[None, :], yout[None, :], h
        print(x.shape, h.shape)

        import plotly.graph_objects as go

        if plot_type.upper() == "MESH3D":
            print(x.shape, y.shape, h.shape)
            print(np.concatenate([x, y, h], axis=1))
            mesh, lines = plotly_triangular_mesh(
                np.concatenate([x, y, -h], axis=0), TRI, showscale=True
            )
            fig = go.Figure(data=[mesh, lines])

        else:
            x = np.linspace(np.min(self.X), np.max(self.X), 101)
            y = np.linspace(np.min(self.Y), np.max(self.Y), 101)
            X, Y = np.meshgrid(x, y)
            h = -self.h_func(X, Y)
            fig = go.Figure(
                data=[go.Surface(z=h.T, x=x, y=y, opacity=0.2, colorbar=dict(x=0.75))]
            )
            fig.update_traces(
                contours_z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_z=True,
                )
            )
            fig.update_traces(
                contours_x=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_x=True,
                )
            )
            fig.update_traces(
                contours_y=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_y=True,
                )
            )

        fig.write_html("Topography.html", auto_open=True, include_mathjax="cdn")

    def energy(self, sols):
        # u, v, η = np.split(sols, 3, axis=1)
        # vec = np.concatenate([u, v, η], axis=1).T

        return (sols.T @ self.M_E @ sols.conjugate())[0, 0]

    def energy_breakdown(self, sols):
        from scipy.sparse import diags
        from scipy.sparse import block_diag

        u, v, η = np.split(sols, 3, axis=0)
        ones = np.ones(u.shape)

        u_vec = np.concatenate([u, v], axis=0)
        # ones2 = np.ones(u_vec.shape)
        H_2 = diags(np.sqrt(self.h))
        H_2 = block_diag(2 * [H_2])
        M_u = H_2 @ block_diag(2 * [self.M]) @ H_2
        M_u2 = diags(np.sqrt(self.h)) @ self.M @ diags(np.sqrt(self.h))
        norm_u, norm = ones.T @ M_u2 @ ones, ones.T @ self.M @ ones
        KE = (u_vec.T @ M_u @ u_vec.conjugate()) / norm_u
        PE = (η.T @ self.M @ η.conjugate()) / norm

        return KE, PE


def plotly_surface_mesh(
    vertices,
    z,
    colorscale="Viridis",
    title="",
    showscale=True,
    reversescale=False,
    cvals=[-1, 1],
    x_loc=1.05,
):
    """


    Parameters
    ----------
    vertices : TYPE
        DESCRIPTION.
    colorscale : TYPE, optional
        DESCRIPTION. The default is "Viridis".
    title : TYPE, optional
        DESCRIPTION. The default is ''.
    showscale : TYPE, optional
        DESCRIPTION. The default is True.
    reversescale : TYPE, optional
        DESCRIPTION. The default is False.
    plot_edges : TYPE, optional
        DESCRIPTION. The default is True.
    cvals : TYPE, optional
        DESCRIPTION. The default is [-1, 1].
    x_loc : TYPE, optional
        DESCRIPTION. The default is 1.05.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """

    x, y, f = vertices
    x, y, f = x.T.flatten(), y.T.flatten(), f.T.flatten()

    surface = dict(
        type="surface",
        x=x,
        y=y,
        z=z * np.ones(x.shape),
        colorscale=colorscale,
        reversescale=reversescale,
        surfacecolor=f,
        name=title,
        showscale=showscale,
        cmin=cvals[0],
        cmax=cvals[1],
    )

    if showscale is True:
        surface.update(colorbar=dict(thickness=10, ticklen=4, len=0.75, x=x_loc))
    return surface


def plotly_triangular_mesh(
    vertices,
    faces,
    intensities=None,
    colorscale="Viridis",
    title="",
    showscale=True,
    reversescale=False,
    plot_edges=True,
    cvals=[-1, 1],
    x_loc=1.05,
):
    """


    Parameters
    ----------
    vertices : TYPE
        DESCRIPTION.
    faces : TYPE
        DESCRIPTION.
    intensities : TYPE, optional
        DESCRIPTION. The default is None.
    colorscale : TYPE, optional
        DESCRIPTION. The default is "Viridis".
    title : TYPE, optional
        DESCRIPTION. The default is ''.
    showscale : TYPE, optional
        DESCRIPTION. The default is True.
    reversescale : TYPE, optional
        DESCRIPTION. The default is False.
    plot_edges : TYPE, optional
        DESCRIPTION. The default is True.
    cvals : TYPE, optional
        DESCRIPTION. The default is [-1, 1].
    x_loc : TYPE, optional
        DESCRIPTION. The default is 1.05.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """

    x, y, z = vertices
    x, y, z = x.T.flatten(), y.T.flatten(), z.T.flatten()
    vertices = np.array([x, y, z])
    I, J, K = faces.T
    if intensities is None:
        intensities = z.T.flatten()

    if hasattr(intensities, "__call__"):
        intensity = z  # the intensities are computed here via the passed function,
        # that returns a list of vertices intensities

    elif isinstance(intensities, (list, np.ndarray)):
        intensity = intensities  # intensities are given in a list
    else:
        raise ValueError("intensities can be either a function or a list, np.array")

    mesh = dict(
        type="mesh3d",
        x=x,
        y=y,
        z=z,
        colorscale=colorscale,
        reversescale=reversescale,
        intensity=intensity,
        i=I,
        j=J,
        k=K,
        name=title,
        showscale=showscale,
        cmin=cvals[0],
        cmax=cvals[1],
    )

    if showscale is True:
        mesh.update(colorbar=dict(thickness=10, ticklen=4, len=0.75, x=x_loc))

    if plot_edges is False:  # the triangle sides are not plotted
        return [mesh]
    else:  # plot edges
        # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        # None separates data corresponding to two consecutive triangles
        tri_vertices = vertices.T[faces]
        Xe, Ye, Ze = [], [], []

        for T in tri_vertices:
            Xe += [T[k % 3][0] for k in range(4)] + [None]
            Ye += [T[k % 3][1] for k in range(4)] + [None]
            Ze += [T[k % 3][2] for k in range(4)] + [None]

        # define the lines to be plotted
        lines = dict(
            type="scatter3d",
            x=Xe,
            y=Ye,
            z=Ze,
            mode="lines",
            name=title,
            line=dict(color="rgb(70,70,70)", width=1, cmin=cvals[0], cmax=cvals[1]),
        )

        return [mesh, lines]


def grid_convert(vals, old_grid, new_grid):
    from scipy.interpolate import griddata

    new_vals = []
    for val in vals:
        val_r = griddata(old_grid, val.real, new_grid, method="cubic")
        val_i = griddata(old_grid, val.imag, new_grid, method="cubic")
        new_vals.append((val_r + 1j * val_i)[:, :, 0])

    return new_vals


if __name__ == "__main__":
    pass
