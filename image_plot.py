from traits.api import HasTraits, Instance
from chaco.api import Plot, ArrayPlotData, TransformColorMapper, jet


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
        self.plot.request_redraw()

    def get_plot_component(self):
        self.plotdata.set_data("imagedata", self.model.Z)
        tcm = TransformColorMapper.from_color_map(jet)
        renderer = self.plot.img_plot("imagedata", colormap=tcm)[0]
        return self.plot
