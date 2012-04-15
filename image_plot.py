from traits.api import HasTraits, Instance
from chaco.api import Plot, ArrayPlotData, TransformColorMapper, LinearMapper, PlotAxis, jet


class ImagePlot(HasTraits):
    """A color image plot
    """
    plot = Instance(Plot)
    plotdata = Instance(ArrayPlotData, ())

    def __init__(self, model):
        super(ImagePlot, self).__init__(model=model)

    def _plot_default(self):
        plot = Plot(self.plotdata)
        return plot

    def update_plotdata(self):
        self.plotdata.set_data("imagedata", self.model.Z)

    def clear_plot(self):
        print 'clearing plot'
        for k in self.plot.plots.keys():
            self.plot.delplot(k)
        self.plot.datasources.clear()
        self.get_plot_component()

    def get_plot_component(self):
        xbounds = self.model.xh[0] / 1000, self.model.xh[-1] / 1000
        ybounds = self.model.yh[0] / 1000, self.model.yh[-1] / 1000
        self.plotdata.set_data("imagedata", self.model.Z)
        #self.plot.x_mapper.stretch_data = False
        #self.plot.y_mapper.stretch_data = False
        cmap = TransformColorMapper.from_color_map(jet)
        renderer = self.plot.img_plot("imagedata",
                                      xbounds = xbounds,
                                      ybounds = ybounds,
                                      colormap=cmap)[0]
        left = PlotAxis(orientation='left',
                        title='km',
                        mapper=self.plot.value_mapper,
                        component=self.plot)
        bottom = PlotAxis(orientation='bottom',
                        title='km',
                        mapper=self.plot.value_mapper,
                        component=self.plot)
        self.plot.underlays.append(left)
        self.plot.underlays.append(bottom)
        return self.plot
