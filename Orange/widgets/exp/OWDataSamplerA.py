import sys
import numpy

import Orange.data
from Orange.widgets import widget, gui
from Orange.widgets.utils.signals import Input, Output


class OWDataSamplerA(widget.OWWidget):
    name = "Data Sampler"
    description = "Randomly select a subset of numbers"
    want_main_area = False

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        sample = Output("Sampled Data", Orange.data.Table)

    def __init__(self):
        super().__init__()

        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, "No data yet")
        self.infob = gui.widgetLabel(box, '')

    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.infoa.setText("%d instances in input data set" % len(dataset))
            indices = numpy.random.permutation(len(dataset))
            indices = indices[:int(numpy.ceil(len(dataset) * 0.1))]
            sample = dataset[indices]
            self.infob.setText("%d sampled instances" % len(sample))
            self.Outputs.sample.send(sample)
        else:
            self.infoa.setText(
                "No data on input yet, waiting to get something.")
            self.infob.setText('')
            self.Outputs.sample.send(None)
