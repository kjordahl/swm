#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Sun May 13 16:04:36 EDT 2012>
"""

import time
import threading
import numpy as np
from numpy import pi, sin, cos, sqrt, exp
from traits.api import (HasTraits, Int, Float, Bool, Enum, Str,
                        List, Range, Array)
from chaco.api import Plot, ArrayPlotData, TransformColorMapper, jet
from scipy import sparse
from scipy.sparse import linalg
from scipy.io.netcdf import netcdf_file
from image_plot import ImagePlot

class ShallowWaterModel(HasTraits):
    """
    Shallow Water Model
    """
    # constants
    a = Float(6370e3)               # (m) Earth's radius
    Omega = Float(2*pi/(24*60**2))  # (1/s) rotational frequency of the Earth
    Ah = Float(1e5)                 # (m**2/s) eddy viscosity
    rho0 = Float(1000)              # (kg/m**3) density
    g = Float(9.81)                 # (m/s**2) acceleration of gravity
    # Parameters
    nx = Int(151)           # number of grid points in the x-direction
    ny = Int(151)           # number of grid points in the y-direction
    Rd = Float(100e3)       # (m) Rossby Radius
    Lx = Float(1000e3)      # (m) East-West domain size
    Ly = Float(1000e3)      # (m) North-South domain size
    lat = Range(low=-90, high=90, value=0)  # (degrees) Reference latitude
    H0 = Int(100)            # (m) reference thickness
    wind_x = Float(0)
    wind_y = Float(0)
    f0 = Float(0)
    beta = Float(0)
    H = Array               # depth grid
    # model
    mask_list = List(Str)
    mask_shape = Enum(values='mask_list')
    running = Bool(False)
    delay = Float(0.0)                 # run loop delay (seconds)
    run_text = Str("Start")
    t = Float(0)                        # run time (seconds)

    def __init__(self):
        self.update_params()
        #self.Z = np.zeros((self.nx, self.ny))
        self.setup_mesh()
        self.initial_conditions()
        #self.operators()
        #self.initialize_matrix()

    def set_plot(self, plot=None):
        self.plot = plot

    def _mask_list_default(self):
        return ['rectangular',
                'periodic',
                'east-west channel',
                'north-south channel',
                'Lake Superior',
                'Gulf of Mexico',
                'Pacific']

    def initial_conditions(self):
        """Geostrophic adjustment problem
        initial condition
        """
        Xbump = self.Lx / 4
        Ybump = self.Ly / 4
        Lbump = 1000.0
        self.h0 = 10 * exp(-((self.Xh - Xbump)**2 + (self.Yh - Ybump)**2) /
                      (Lbump)**2)
        self.Z = self.h0
        self.Z[self.msk==0] = np.nan
        self.u0 = np.zeros(self.Xv.shape)
        self.v0 = np.zeros(self.Yv.shape)
        self.t = 0

    def _mask_shape_changed(self):
        self.set_mask()
        self.Xbump = self.Lx
        self.Ybump = self.Ly / 2
        self.update_params()
        self.setup_mesh()
        self.initial_conditions()
        if hasattr(self, 'plot'):
            self.plot.clear_plot()
            self.plot.update_plotdata()

    def _running_changed(self):
        if self.running:
            self.update_params()
            self.setup_mesh()
            self.initial_conditions()
            self.operators()
            self.initialize_matrix()
            self.start()
            self.run_text = 'Stop'
        else:
            self.run_text = 'Restart'

    def d0(self, M):
        m = M.flatten()
        n = len(m)
        return sparse.spdiags(m, 0, n, n)

    def update_params(self):
        """update calculated parameters"""
        self.dx = 1.0 * self.Lx / self.nx
        self.dy = 1.0 * self.Ly / self.ny
        self.phi0 = pi * self.lat / 180           # reference latitude (radians)
        self.gp = 10.0
        self.cg = sqrt(self.gp * self.H0)

    def setup_mesh(self):
        self.set_mask()
        dx = self.dx
        dy = self.dy
        # mesh for the h-points
        xh = np.arange(dx / 2, self.Lx, dx)
        yh = np.arange(dy / 2, self.Ly, dy)
        self.Xh, self.Yh = np.meshgrid(xh, yh)
        # mesh for the u-points
        xu = np.arange(dx, self.Lx + dx, dx)
        yu = yh
        self.Xu, self.Yu = np.meshgrid(xu, yu)
        # mesh for the v-points
        xv = xh
        yv = np.arange(dy, self.Ly + dy, dy)
        self.Xv, self.Yv = np.meshgrid(xv, yv)
        self.xh = xh
        self.yh = yh

    def set_mask(self):
        """Land-sea mask defined on the h-points
        1 = ocean point
        0 = land point
        """
        self.msk = np.ones((self.ny, self.nx))
        if self.mask_shape == 'rectangular':
            self.msk[:,-1] = 0
            self.msk[-1,:] = 0
            self.H = self.H0 * np.ones(self.msk.shape)
        elif self.mask_shape == 'east-west channel':
            self.msk[-1,:] = 0
            self.H = self.H0 * np.ones(self.msk.shape)
        elif self.mask_shape == 'north-south channel':
            self.msk[:,-1] = 0
            self.H = self.H0 * np.ones(self.msk.shape)
        elif self.mask_shape == 'periodic':
            self.H = self.H0 * np.ones(self.msk.shape)
        elif self.mask_shape == 'Lake Superior':
            self.nx = 151
            self.ny = 151
            self.Lx = 600e3
            self.Ly = 600e3
            self.lat = 43                   # Latitude of Lake Superior
            self.H0 = 150
            n = netcdf_file('superior_mask.grd', 'r')
            z = n.variables['z']
            self.msk = z.data
        elif self.mask_shape == 'Gulf of Mexico':
            self.load_grid('gulf')
            self.Lx = 2000e3
            self.Ly = 1300e3
            self.lat = 25
            self.H0 = 1600
        elif self.mask_shape == 'Pacific':
            self.load_grid('pacific')
            self.lat = 10               # doesn't work right at equator?
            self.Lx = 20000e3
            self.Ly = 25000e3
            self.H0 = 1000

    def load_grid(self, name):
        """Load boundary conditions from a netcdf (GMT) grid file"""
        n = netcdf_file(name + '.grd', 'r')
        z = n.variables['z']
        self.H = np.array(z.data)
        self.H[self.H < 1.0] = 0
        self.msk = np.ones(self.H.shape)
        self.msk[self.H==0] = 0
        self.ny, self.nx = self.H.shape

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
        hDIVu = ((I - IW) * self.d0(self.msk) *
                self.d0(IE * self.msk.flatten())) / self.dx
        hDIVv = ((I - IS) * self.d0(self.msk) *
                 self.d0(IN * self.msk.flatten())) / self.dy
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
        fu = self.f0 + self.beta * self.Yu
        fv = self.f0 + self.beta * self.Yv

        # Linear swm operator
        self.L = sparse.vstack([sparse.hstack([-self.Ah * DEL2u,
                                               -self.d0(fu) * uAv,
                                               self.gp * DX]),
                                sparse.hstack([self.d0(fv) * vAu,
                                               -self.Ah * DEL2v,
                                               self.gp * DY]),
                                sparse.hstack([self.H0 * hDIVu,
                                               self.H0 * hDIVv,
                                               sparse.csc_matrix((n, n))])]).tocsc()
        self.IE = IE
        self.IN = IN

    def initialize_matrix(self):
        """Set up the state vector, matrix, and index variables
        Pre-factor the matrix for efficiency in the time loop
        """
        n = self.nx * self.ny
        # keep only points where u is not 0
        ukeep = self.msk.flatten() * (self.IE * self.msk.flatten())
        # keep only points where v is not 0
        vkeep = self.msk.flatten() * (self.IN * self.msk.flatten())
        hkeep = self.msk.flatten()
        keep = np.hstack([ukeep, vkeep, hkeep])
        ikeep = np.nonzero(keep)[0]
        self.ikeep = ikeep

        #self.sbig = self.s
        self.s = self.sbig[np.nonzero(keep)]
        # indices of ocean points in the 2-d fields
        self.ih = np.nonzero(hkeep)
        self.iu = np.nonzero(ukeep)
        self.iv = np.nonzero(vkeep)
        # indices of variables inside the big s vector
        self.iubig = np.nonzero(np.hstack([ukeep,
                                           np.zeros(vkeep.shape),
                                           np.zeros(hkeep.shape)]))
        self.ivbig = np.nonzero(np.hstack([np.zeros(ukeep.shape),
                                           vkeep,
                                           np.zeros(hkeep.shape)]))
        self.ihbig = np.nonzero(np.hstack([np.zeros(ukeep.shape),
                                           np.zeros(vkeep.shape),
                                           hkeep]))
        dt = 0.5 * self.dx / self.cg
        I = sparse.eye(3*n, 3*n).tocsc()
        A = I + (dt / 2) * self.L
        B = I - (dt / 2) * self.L
        A = A[ikeep, :]
        A = A[:, ikeep]                  # does this get used?
        B = B[ikeep, :]
        self.B = B[:, ikeep]
        self.dt = dt

        print 'Factoring the big matrix...',
        tic = time.time()
        self.solve = linalg.factorized(A)
        print 'Elapsed time: ', time.time() - tic
        self.h = np.zeros(self.msk.shape).flatten()
        self.u = np.zeros(self.msk.shape).flatten()
        self.v = np.zeros(self.msk.shape).flatten()
        self.V = self.v.reshape(self.msk.shape)
        self.U = self.u.reshape(self.msk.shape)
        self.Z = self.h.reshape(self.msk.shape)

    def body_forces(self):
        """Update body forces from wind stress vector"""
        wind_speed = sqrt(self.wind_x ** 2 + self.wind_y ** 2)
        b = np.zeros(self.Xh.shape)          # bouyancy term
        tau = 1.4331e-4 * wind_speed ** 3
        theta = np.arctan2(self.wind_y, self.wind_x)
        tau_x = tau * cos(theta) * np.ones(self.Xu.shape)
        tau_y = tau * sin(theta) * np.ones(self.Yu.shape)
        F = np.hstack([tau_x.flatten() / (self.rho0 * self.H0),
                          tau_y.flatten() / (self.rho0 * self.H0),
                          b.flatten()])
        return F[self.ikeep]

    def time_step(self):
        """Update state vector and height and velocity fields at each time step
        """
        F = self.body_forces()
        self.s = self.solve(self.B * self.s + self.dt * F)
        self.sbig[self.ikeep] = self.s
        self.u[self.iu] = self.sbig[self.iubig]
        self.v[self.iv] = self.sbig[self.ivbig]
        self.h[self.ih] = self.sbig[self.ihbig]
        self.V = self.v.reshape(self.msk.shape)
        self.U = self.u.reshape(self.msk.shape)
        self.Z = self.h.reshape(self.msk.shape)
        self.Z[self.msk==0] = np.nan
        self.t = self.t + self.dt

    def start(self):
        """Start a thread to run the time steps"""
        self.running = True
        thread = threading.Thread(target=run_loop, args=(self,))
        thread.start()

    def stop(self):
        self.running = False


def run_loop(model):
    tic = time.time()
    while model.running:
        #print time.time() - tic
        tic = time.time()
        model.time_step()
        model.plot.plotdata.set_data("imagedata", model.Z)
        time.sleep(model.delay)


def main():
    swm = ShallowWaterModel()
    plot = ImagePlot(swm)
    swm.set_plot(plot)

    import enaml
    with enaml.imports():
        from swm_view import SimpleView
    view = SimpleView(model=swm, plot=plot)

    view.show()

if __name__ == '__main__':
    main()
