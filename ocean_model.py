#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Thu Apr 19 17:00:19 EDT 2012>
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
    Xbump = Float
    Ybump = Float
    Lbump = Range(low=0.0, high=10.0, value=1.0)  # size of bump (relative to Rd)
    L0 = Float(100)                               # height of bump
    lat = Range(low=-90, high=90, value=30)       # (degrees) Reference latitude
    H0 = Int(600)            # (m) reference thickness
    wind_x = Float(0)

    def __init__(self):
        self.Xbump = self.Lx
        self.Ybump = self.Ly / 2
        super(OceanModel, self).__init__()

    def initial_conditions(self):
        """Geostrophic adjustment problem
        initial condition
        """
        super(OceanModel, self).initial_conditions()
        if self.Lbump * self.Rd > 0:
            self.h0 = self.L0 * exp(-((self.Xh - self.Xbump)**2 +
                                      (self.Yh - self.Ybump)**2) /
                                (self.Lbump * self.Rd)**2)
        else:
            self.h0 = np.zeros(self.msk.shape)
        self.Z = self.h0
        self.Z[self.msk==0] = np.nan

    def update_params(self):
        """set rotational parameters"""
        super(OceanModel, self).update_params()
        self.f0 = 2 * self.Omega * sin(self.phi0)  # (1/s) Coriolis parameter
        self.beta = (2 * self.Omega / self.a) * cos(self.phi0) # (1/(ms))
        if self.f0 == 0:
            self.Ah = 0.01 * self.dx**2 / 8.64e4
            self.gp = self.Rd**2 * self.beta / self.H0 # (m/s^2) reduced gravity
        else:
            self.Ah = 0.01 * self.dx**2 / (2*pi/self.f0)
            self.gp = (self.f0 * self.Rd)**2 / self.H0 # (m/s^2) reduced gravity
        self.cg = sqrt(self.gp * self.H0)


    def _Lbump_changed(self):
        self.set_mask()
        self.update_params()
        self.setup_mesh()
        self.initial_conditions()
        if hasattr(self, 'plot'):
            self.plot.update_plotdata()

    def _Xbump_changed(self):
        self.set_mask()
        self.update_params()
        self.setup_mesh()
        self.initial_conditions()
        if hasattr(self, 'plot'):
            self.plot.update_plotdata()


def run_loop(model):
    tic = time.time()
    while model.running:
        model.time_step()
        model.plot.update_plotdata()
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
