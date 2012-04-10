#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Tue Apr 10 08:44:50 EDT 2012>
"""

from scipy.io.netcdf import netcdf_file
from ocean_model import ShallowWaterModel, OceanPlot
from traits.api import Int

class WindDrivenModel(ShallowWaterModel):
    """Class for wind driven model

    Set flat initial conditions on Lake Superior
    """

    def __init__(self):
        self.nx = 151
        self.ny = 151
        self.Lbump = 0.0
        self.Lx = 600e3
        self.Ly = 600e3
        self.lat = 43                   # Latitude of Lake Superior
        super(WindDrivenModel, self).__init__()

    def set_mask(self):
        n = netcdf_file('superior_mask.grd', 'r')
        z = n.variables['z']
        self.msk = z.data


def main():
    swm = WindDrivenModel()
    plot = OceanPlot(swm)
    swm.set_plot(plot)

    import enaml
    with enaml.imports():
        from wind_view import WindView
    view = WindView(model=swm, plot=plot)

    view.show()

if __name__ == '__main__':
    main()
