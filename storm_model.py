#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Sun Apr 15 20:52:48 EDT 2012>
"""

import numpy as np
from numpy import pi, sin, cos, sqrt, exp
from scipy.io.netcdf import netcdf_file
from ocean_model import OceanModel
from image_plot import ImagePlot


class StormModel(OceanModel):
    """Wind stress for a stationary hurricane in the Gulf of Mexico
    Hurricane Katrina at 0600Z 29 Aug 2005 (shortly before landfall)
    based on model data at
    http://www.aoml.noaa.gov/hrd/Storm_pages/katrina2005/wind.html
    """

    def __init__(self):
        self.mask_shape = 'Gulf of Mexico'
        super(StormModel, self).__init__()
        self.Lbump = 0.0
        n = netcdf_file('wind_x.grd', 'r')
        wind_x = n.variables['z'].data
        n = netcdf_file('wind_y.grd', 'r')
        wind_y = n.variables['z'].data
        tau_mag = 1.4331e-4 * (wind_x ** 2 * wind_y ** 2) ** (3 / 2)
        theta = np.arctan2(wind_y, wind_x)
        self.tau_x = tau_mag * cos(theta)
        self.tau_y = tau_mag * sin(theta)

    def body_forces(self):
        """Update body forces from wind stress vector"""
        b = np.zeros(self.Xh.shape)          # bouyancy term
        F = np.hstack([self.tau_x.flatten() / (self.rho0 * self.H),
                          self.tau_y.flatten() / (self.rho0 * self.H),
                          b.flatten()])
        return F[self.ikeep]


def main():
    swm = StormModel()
    plot = ImagePlot(swm)
    swm.set_plot(plot)

    import enaml
    with enaml.imports():
        from storm_view import StormView
    view = StormView(model=swm, plot=plot)

    view.show()

if __name__ == '__main__':
    main()
