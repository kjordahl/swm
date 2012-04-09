#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Mon Apr  9 18:17:23 EDT 2012>
"""

from ocean_model import ShallowWaterModel, OceanPlot

class WindDrivenModel(ShallowWaterModel):
    """Class for wind driven model

    Just set the initial condition to flat."""

    def __init__(self):
        self.Lbump = 0.0
        super(WindDrivenModel, self).__init__()


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
