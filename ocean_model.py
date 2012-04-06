#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Fri Apr  6 08:12:31 EDT 2012>
"""

import time
import threading
import numpy as np
from numpy import pi, sin, cos, sqrt, exp
from traits.api import (HasTraits, Int, Float, Instance, Bool, Enum, Str,
                        Range, on_trait_change)
from chaco.api import Plot, ArrayPlotData
from scipy import sparse
from scipy.sparse import linalg


class ShallowWaterModel(HasTraits):
    """
    Shallow Water Model
    """
    # constants
    a = Float(6370e3)               # (m) Earth's radius
    Omega = Float(2*pi/(24*60**2))  # (1/s) rotational frequency of the Earth
    Ah = Float(1e4)         # (m^2/s) viscosity
    # Parameters
    nx = Int(101)           # number of grid points in the x-direction
    ny = Int(101)           # number of grid points in the y-direction
    Rd = Float(100e3)       # (m) Rossby Radius
    Lx = Float(1200e3)      # (m) East-West domain size
    Ly = Float(1200e3)      # (m) North-South domain size
    Xbump = Range(low=-1.0, high=1.0, value=0.0)
    Lbump = Range(low=0.1, high=10.0, value=1.0)        # size of bump (relative to Rd)
    lat = Int(-30)           # (degrees) Reference latitude
    H = Int(600)            # (m) reference thickness
    # model
    running = Bool(False)
    delay = Float(0.01)    # run loop delay (seconds)
    run_text = Str("Start")

    def __init__(self):
        self.update_params()
        self.Z = np.zeros((self.nx, self.ny))
        #self.setup_mesh()
        #self.initial_conditions()
        #self.operators()
        #self.initialize_matrix()

    def set_plot(self, plot=None):
        self.plot = plot

    def initial_conditions(self):
        """Geostrophic adjustment problem
        initial condition
        """
        Xbump = (self.Xbump + 1.0) * self.Ly / 2
        Ybump = self.Ly / 2
        self.h0 = exp(-((self.Xh-Xbump)**2+(self.Yh-Ybump)**2)/(self.Lbump*self.Rd)**2)
        self.Z = self.h0
        self.u0 = np.zeros(self.Xv.shape)
        self.v0 = np.zeros(self.Yv.shape)

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
        n = len(m);
        return sparse.spdiags(m,0,n,n);

    def update_params(self):
        """update calculated parameters"""
        dx = self.Lx / self.nx
        dy = self.Ly / self.ny
        self.phi0 = pi*self.lat/180            # reference latitude (radians)
        self.f0 = 2*self.Omega*sin(self.phi0)  # (1/s) Coriolis parameter
        self.beta = (2*self.Omega/self.a)*cos(self.phi0) # (1/(ms))
        if self.f0 == 0:
            self.Ah = 0.01 * dx**2 / 8.64e4
            self.gp = self.Rd**2 * self.beta / self.H # (m/s^2) reduced gravity
        else:
            self.Ah = 0.01 * dx**2 / (2*pi/self.f0)
            self.gp = (self.f0 * self.Rd)**2 / self.H # (m/s^2) reduced gravity
        self.cg = sqrt(self.gp*self.H)

    def setup_mesh(self):
        dx = self.Lx / self.nx
        dy = self.Ly / self.ny
        # mesh for the h-points
        xh = np.arange(dx/2, self.Lx, dx)
        yh = np.arange(dy/2, self.Ly, dy)
        self.Xh, self.Yh = np.meshgrid(xh, yh)
        # mesh for the u-points
        self.xu = np.arange(dx, self.Lx + dx, dx)
        self.yu = yh
        self.Xu, self.Yu = np.meshgrid(self.xu, self.yu)
        # mesh for the v-points
        xv = xh
        yv = np.arange(dy, self.Ly + dy, dy)
        self.Xv, self.Yv = np.meshgrid(xv,yv);
        # Land-sea mask defined on the h-points
        # 1 = ocean point
        # 0 = land point
        self.msk = np.ones((self.ny, self.nx))
        self.msk[:,-1] = 0
        self.msk[-1,:] = 0
        self.dx = dx
        self.dy = dy

    def operators(self):
        # Differential operators
        n = self.nx*self.ny
        I = sparse.eye(n, n).tocsc()
        ii = np.arange(n).reshape(self.nx, self.ny, order='F')
        ie = np.roll(ii, -1, 1)
        iw = np.roll(ii, 1, 1)
        iin = np.roll(ii, -1, 0)            # "in" is a reserved word
        iis = np.roll(ii, 1, 0)             # so is "is"
        IE = I[ie.flatten('F'),:n]
        IW = I[iw.flatten('F'),:n]
        IN = I[iin.flatten('F'),:n]
        IS = I[iis.flatten('F'),:n]

        DX = (1/self.dx)*(IE-I)
        DY = (1/self.dy)*(IN-I)
        GRAD = sparse.hstack([DX, DY])

        DIV = (1/(self.dx*self.dy))*sparse.hstack([I*self.dy-IW*self.dy, I*self.dx-IS*self.dx])
        hDIV = (1/(self.dx*self.dy))*sparse.hstack([self.dy*(I-IW)*self.d0(self.msk)*self.d0(IE*self.msk.flatten()),self.dx*(I-IS)*self.d0(self.msk)*self.d0(IN*self.msk.flatten())])
        # GRAD for the case of no slip boundary conditions
        # DEL2 for the v points
        # GRAD that assumes that v is zero on the boundary
        DX0 = self.d0(self.msk)*self.d0(IE*self.msk.flatten())*DX+self.d0(self.msk)*self.d0(1-IE*self.msk.flatten())*((1/self.dx)*(-2*I))+self.d0(1-self.msk)*self.d0(IE*self.msk.flatten())*((1/self.dx)*(2*IE))
        DY0 = DY
        GRADv = sparse.vstack([DX0, DY0])
        DEL2v = DIV*GRADv
        # DEL2 for the u ponts
        # GRAD that assumes that u is zero on the boundary
        DX0 = DX
        DY0 = self.d0(self.msk)*self.d0(IN*self.msk.flatten())*DY+self.d0(self.msk)*self.d0(1-IN*self.msk.flatten())*((1/self.dy)*(-2*I))+self.d0(1-self.msk)*self.d0(IN*self.msk.flatten())*((1/self.dy)*(2*IN))
        GRADu = sparse.vstack([DX0, DY0])
        DEL2u = DIV*GRADu
        # Averaging operators that zero out the velocities through the boundaries
        uAv = 0.25*(I+IE+IS+IS*IE)*self.d0(self.msk)*self.d0(IN*self.msk.flatten())
        vAu = 0.25*(I+IN+IW+IN*IW)*self.d0(self.msk)*self.d0(IE*self.msk.flatten())
        # State vector
        self.sbig = np.hstack([self.u0.flatten(), self.v0.flatten(), self.h0.flatten()])

        fu = self.f0 + self.beta * self.Yu
        fv = self.f0 + self.beta * self.Yv

        # Linear swm operator
        self.L = sparse.vstack([sparse.hstack([-self.Ah*DEL2u, -self.d0(fu)*uAv, self.gp*DX]),
                           sparse.hstack([self.d0(fv)*vAu, -self.Ah*DEL2v, self.gp*DY]),
            sparse.hstack([self.H*hDIV, sparse.csc_matrix((n,n))])]).tocsc()
        self.IE = IE
        self.IN = IN

    def initialize_matrix(self):
        """Set up the state vector, matrix, and index variables
        Pre-factor the matrix for efficiency in the time loop
        """
        n = self.nx * self.ny
        ukeep = self.msk.flatten()*self.IE*self.msk.flatten() # keep only pnts where u not 0 
        vkeep = self.msk.flatten()*self.IN*self.msk.flatten() # keep only pnts where v not 0
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
        self.iubig = np.nonzero(np.hstack([ukeep, np.zeros(vkeep.shape), np.zeros(hkeep.shape)]))
        self.ivbig = np.nonzero(np.hstack([np.zeros(ukeep.shape), vkeep, np.zeros(hkeep.shape)]))
        self.ihbig = np.nonzero(np.hstack([np.zeros(ukeep.shape), np.zeros(vkeep.shape), hkeep]))

        dt = 0.5 * self.dx / self.cg
        I = sparse.eye(3*n, 3*n).tocsc()
        A = I + (dt/2) * self.L
        B = I - (dt/2) * self.L
        A = A[ikeep,:]
        A = A[:,ikeep]                  # does this get used?
        B = B[ikeep,:]
        self.B = B[:,ikeep]

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

    def time_step(self):
        """Update state vector and height and velocity fields at each time step
        """
        self.s = self.solve(self.B * self.s)
        self.sbig[self.ikeep] = self.s
        self.u[self.iu] = self.sbig[self.iubig]
        self.v[self.iv] = self.sbig[self.ivbig]
        self.h[self.ih] = self.sbig[self.ihbig]
        self.V = self.v.reshape(self.msk.shape)
        self.U = self.u.reshape(self.msk.shape)
        self.Z = self.h.reshape(self.msk.shape)
        self.Z[self.msk==0] = np.nan

    def start(self):
        """Start a thread to run the time steps"""
        self.running = True
        thread = threading.Thread(target=run_loop, args=(self,))
        thread.start()

    def stop(self):
        self.running = False


class OceanPlot(HasTraits):
    plot = Instance(Plot)
    plotdata = Instance(ArrayPlotData, ())

    def __init__(self, model):
        super(OceanPlot, self).__init__(model=model)

    def _plot_default(self):
        plot = Plot(self.plotdata)
        return plot

    def clear_plot(self):
        print 'clearing plot'
        for k in self.plot.plots.keys():
            self.plot.delplot(k)
        self.plot.datasources.clear()
        self.plot.request_redraw()

    def get_plot_component(self):
        self.plotdata.set_data("imagedata", self.model.Z)
        renderer = self.plot.img_plot("imagedata")[0]
        return self.plot


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
    plot = OceanPlot(swm)
    swm.set_plot(plot)

    import enaml
    with enaml.imports():
        from ocean_view import OceanView
    view = OceanView(model=swm, plot=plot)

    view.show()

if __name__ == '__main__':
    main()
