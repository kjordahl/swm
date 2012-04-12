#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Wed Apr 11 20:11:41 EDT 2012>
"""

import time
import threading
import numpy as np
from numpy import pi, sin, cos, sqrt, exp
from traits.api import (HasTraits, Int, Float, Instance, Bool, Enum, Str,
                        Range, on_trait_change)
from scipy import sparse
from scipy.sparse import linalg
from shallow_water_model import ShallowWaterModel
from image_plot import ImagePlot


class OceanModel(ShallowWaterModel):
    """
    Ocean scale shallow water model with coriolis effects
    """
    # Parameters
    nx = Int(101)           # number of grid points in the x-direction
    ny = Int(101)           # number of grid points in the y-direction
    Rd = Float(100e3)       # (m) Rossby Radius
    Lx = Float(1200e3)      # (m) East-West domain size
    Ly = Float(1200e3)      # (m) North-South domain size
    Xbump = Range(low=-1.0, high=1.0, value=1.0)
    Lbump = Range(low=0.0, high=10.0, value=1.0)  # size of bump (relative to Rd)
    L0 = Float(100)                               # height of bump
    lat = Range(low=-90, high=90, value=30)       # (degrees) Reference latitude
    H = Int(600)            # (m) reference thickness
    wind_x = Float(0)

    def __init__(self):
        super(OceanModel, self).__init__()

    def initial_conditions(self):
        """Geostrophic adjustment problem
        initial condition
        """
        super(OceanModel, self).initial_conditions()
        Xbump = (self.Xbump + 1.0) * self.Ly / 2
        Ybump = self.Ly / 2
        self.h0 = self.L0 * exp(-((self.Xh - Xbump)**2 + (self.Yh - Ybump)**2) /
                      (self.Lbump * self.Rd)**2)
        self.Z = self.h0
        self.Z[self.msk==0] = np.nan

    def update_params(self):
        """set rotational parameters"""
        super(OceanModel, self).update_params()
        self.f0 = 2 * self.Omega * sin(self.phi0)  # (1/s) Coriolis parameter
        self.beta = (2 * self.Omega / self.a) * cos(self.phi0) # (1/(ms))
        if self.f0 == 0:
            self.Ah = 0.01 * self.dx**2 / 8.64e4
            self.gp = self.Rd**2 * self.beta / self.H # (m/s^2) reduced gravity
        else:
            self.Ah = 0.01 * self.dx**2 / (2*pi/self.f0)
            self.gp = (self.f0 * self.Rd)**2 / self.H # (m/s^2) reduced gravity
        print 'gp', self.gp
        self.cg = sqrt(self.gp * self.H)


    def _Lbump_changed(self):
        self.setup_mesh()
        self.initial_conditions()
        if hasattr(self, 'plot'):
            self.plot.update_plotdata()

    def _Xbump_changed(self):
        self.setup_mesh()
        self.initial_conditions()
        if hasattr(self, 'plot'):
            self.plot.update_plotdata()


def run_loop(model):
    tic = time.time()
    while model.running:
        #print time.time() - tic
        tic = time.time()
        model.time_step()
        model.plot.plotdata.set_data("imagedata", model.Z)
        time.sleep(model.delay)


def main():
    swm = OceanModel()
    plot = ImagePlot(swm)
    swm.set_plot(plot)

    import enaml
    with enaml.imports():
        from ocean_view import OceanView
    view = OceanView(model=swm, plot=plot)

    view.show()

if __name__ == '__main__':
    main()
