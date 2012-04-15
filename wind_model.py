#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Sun Apr 15 15:35:37 EDT 2012>
"""

from scipy.io.netcdf import netcdf_file
from ocean_model import OceanModel
from image_plot import ImagePlot
from traits.api import Int

class WindDrivenModel(OceanModel):
    """Class for wind driven model

    Set flat initial conditions on Lake Superior
    """

    def __init__(self):
        self.mask_shape = 'Lake Superior'
        super(WindDrivenModel, self).__init__()
        self.Lbump = 0.0


def main():
    swm = WindDrivenModel()
    plot = ImagePlot(swm)
    swm.set_plot(plot)

    import enaml
    with enaml.imports():
        from wind_view import WindView
    view = WindView(model=swm, plot=plot)

    view.show()

if __name__ == '__main__':
    main()
