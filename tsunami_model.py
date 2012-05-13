#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Sun May 13 16:33:13 EDT 2012>
"""

from scipy.io.netcdf import netcdf_file
from shallow_water_model import ShallowWaterModel
from ocean_model import OceanModel
from image_plot import ImagePlot
from traits.api import Int
import numpy as np
from numpy import pi, sin, cos, sqrt, exp
from scipy import sparse
from scipy.sparse import linalg
from IPython.frontend.terminal.embed import InteractiveShellEmbed


class TsunamiModel(ShallowWaterModel):
    """Class for depth dependent model

    try on a flat (Mercatorized) Pacific
    """
    roughness = 0.3         # meters ** (-1/3) Manning roughness coefficient
    eps = 100.0             # meters (desingularize velocities as depth less than eps)

    def __init__(self):
        self.nx = 101
        self.ny = 101
        self.Ah = 5e4
        #self.mask_shape = 'periodic'
        self.mask_shape = 'Gulf of Mexico'
        super(TsunamiModel, self).__init__()

    def _mask_list_default(self):
        return ['Gulf of Mexico',
                'Pacific']

    def update_params(self):
        super(TsunamiModel, self).update_params()
        #self.eps = 0.01 * min(self.dx, self.dy) ** 4

    def initialize_matrix(self):
        super(TsunamiModel, self).initialize_matrix()
        corrbig = np.hstack([self.ucorr.flatten(),
                             self.vcorr.flatten(),
                             np.ones(self.h0.shape).flatten()])
        self.corr = corrbig[self.ikeep]

    def initial_conditions(self):
        """Tsunami initial condition
        """
        self.Cf = self.g * self.roughness ** 2 / self.H ** (1/3)
        Xbump = self.Lx / 2
        Ybump = self.Ly / 2
        Lbump = 1000.0
        self.h0 = 10 * exp(-((self.Xh - Xbump)**2 + (self.Yh - Ybump)**2) /
                      (Lbump)**2)
        self.Z = self.h0
        self.Z[self.msk==0] = np.nan
        self.u0 = np.zeros(self.Xv.shape)
        self.v0 = np.zeros(self.Yv.shape)
        self.t = 0

    def operators(self):
        """Define differential operators
        """
        n = self.nx * self.ny
        I = sparse.eye(n, n).tocsc()
        ii = np.arange(n).reshape(self.nx, self.ny, order='F')
        ie = np.roll(ii, -1, 1)
        iw = np.roll(ii, 1, 1)
        iin = np.roll(ii, -1, 0)            # "in" is a reserved word
        iis = np.roll(ii, 1, 0)             # so is "is"
        IE = I[ie.flatten('F'), :n]
        IW = I[iw.flatten('F'), :n]
        IN = I[iin.flatten('F'), :n]
        IS = I[iis.flatten('F'), :n]

        DX = (1 / self.dx) * (IE - I)
        DY = (1 / self.dy) * (IN - I)
        GRAD = sparse.hstack([DX, DY])

        DIV = ((1 / (self.dx * self.dy)) *
               sparse.hstack([I * self.dy - IW * self.dy,
                              I * self.dx - IS * self.dx]))
        Hu = (self.H + np.roll(self.H, 1, 1)) / 2
        self.ucorr = np.sqrt(2) * Hu ** 2 / np.sqrt(Hu ** 4 + np.maximum(self.eps, Hu ** 4))
        hDIVu = ((I - IW) * self.d0(Hu) *
                self.d0(IE * self.msk.flatten())) / self.dx
        Hv = (self.H + np.roll(self.H, 1, 0)) / 2
        self.vcorr = np.sqrt(2) * Hv ** 2 / np.sqrt(Hv ** 4 + np.maximum(self.eps, Hv ** 4))

        hDIVv = ((I - IS) * self.d0(Hv) *
                 self.d0(IN * self.msk.flatten())) / self.dy
        ix = range(self.nx)
        ix.append(0)
        #ix.insert(0, self.nx - 1)
        iy = range(self.ny)
        iy.append(0)
        #iy.insert(0, self.ny - 1)
        dHx = np.diff(self.H[:,ix], axis=1) / self.dx
        dHy = np.diff(self.H[iy,:], axis=0) / self.dy
        # GRAD for the case of no slip boundary conditions
        # DEL2 for the v points
        # GRAD that assumes that v is zero on the boundary
        DXv = (self.d0(self.msk) * self.d0(IE * self.msk.flatten()) * DX +
               self.d0(self.msk) * self.d0(1 - IE * self.msk.flatten()) *
               ((1 / self.dx) * (-2 * I)) + self.d0(1 - self.msk) *
               self.d0(IE * self.msk.flatten()) * ((1 / self.dx) * (2 * IE)))
        DYv = DY
        GRADv = sparse.vstack([DXv, DYv])
        DEL2v = DIV * GRADv
        # DEL2 for the u ponts
        # GRAD that assumes that u is zero on the boundary
        DXu = DX
        DYu = (self.d0(self.msk) * self.d0(IN * self.msk.flatten()) * DY +
               self.d0(self.msk) * self.d0(1 - IN * self.msk.flatten()) *
               ((1 / self.dy) * (-2 * I)) + self.d0(1 - self.msk) *
               self.d0(IN * self.msk.flatten()) * ((1 / self.dy) * (2 * IN)))
        GRADu = sparse.vstack([DXu, DYu])
        DEL2u = DIV * GRADu
        # Averaging operators that zero out the velocities through the boundaries
        Ise = 0.25 * (I + IE + IS + IS * IE)
        Inw = 0.25 * (I + IN + IW + IN * IW)
        uAv = Ise * self.d0(self.msk) * self.d0(IN * self.msk.flatten())
        vAu = Inw * self.d0(self.msk) * self.d0(IE * self.msk.flatten())
        # State vector
        self.sbig = np.hstack([self.u0.flatten(),
                               self.v0.flatten(),
                               self.h0.flatten()])
        fu = self.f0 + self.beta * self.Yu * 0
        fv = self.f0 + self.beta * self.Yv * 0
        # Linear swm operator
        self.L = sparse.vstack([sparse.hstack([-self.Ah * DEL2u,
                                               -self.d0(fu) * uAv,
                                               self.gp * DX]),
                                sparse.hstack([self.d0(fv) * vAu,
                                               -self.Ah * DEL2v,
                                               self.gp * DY]),
                                sparse.hstack([self.d0(dHx) + hDIVu,
                                               self.d0(dHy) + hDIVv,
                                               sparse.csc_matrix((n, n))])]).tocsc()
        self.IE = IE
        self.IN = IN

    def body_forces(self):
        """Apply bottom friction as body force"""
        # translate U and V to the H grid
        Uh = (self.U + np.roll(self.U, -1, 1)) / 2
        Vh = (self.V + np.roll(self.V, -1, 0)) / 2
        b = np.zeros(self.Xh.shape)          # bouyancy term
        factor = self.Cf * sqrt(Uh ** 2 + Vh ** 2) / self.H
        tau_x = - Uh * factor
        tau_y = - Vh * factor
        F = np.hstack([tau_x.flatten(),
                       tau_y.flatten(),
                       b.flatten()])
        return F[self.ikeep]


    def time_step(self):
        """Update state vector and height and velocity fields at each time step
        """
        F = self.body_forces()
        self.s = self.solve(self.B * self.s + self.dt * F) * self.corr
        self.sbig[self.ikeep] = self.s
        self.u[self.iu] = self.sbig[self.iubig]
        self.v[self.iv] = self.sbig[self.ivbig]
        self.h[self.ih] = self.sbig[self.ihbig]
        self.V = self.v.reshape(self.msk.shape)
        self.U = self.u.reshape(self.msk.shape)
        self.Z = self.h.reshape(self.msk.shape)
        self.Z[self.msk==0] = np.nan
        self.t = self.t + self.dt


def main():
    swm = TsunamiModel()
    plot = ImagePlot(swm)
    swm.set_plot(plot)

    import enaml
    with enaml.imports():
        from tsunami_view import TsunamiView
    view = TsunamiView(model=swm, plot=plot)

    view.show()

if __name__ == '__main__':
    main()
