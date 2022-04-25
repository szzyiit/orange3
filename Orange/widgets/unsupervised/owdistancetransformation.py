import numpy as np

from Orange.util import scale
from Orange.misc import DistMatrix
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output


class OWDistanceTransformation(widget.OWWidget):
    name = "距离变换(Distance Transformation)"
    description = "根据所选标准变换距离。"
    icon = "icons/DistancesTransformation.svg"
    keywords = ['julibianhuan', 'bianhuan']
    category = '非监督(Unsupervised)'

    class Inputs:
        distances = Input("距离(Distances)", DistMatrix, replaces=['Distances'])

    class Outputs:
        distances = Output("距离(Distances)", DistMatrix, dynamic=False, replaces=['Distances'])

    want_main_area = False
    resizing_enabled = False

    normalization_method = settings.Setting(0)
    inversion_method = settings.Setting(0)
    autocommit = settings.Setting(True)

    normalization_options = (
        ("No normalization", lambda x: x),
        ("To interval [0, 1]", lambda x: scale(x, min=0, max=1)),
        ("To interval [-1, 1]", lambda x: scale(x, min=-1, max=1)),
        ("Sigmoid function: 1/(1+exp(-X))", lambda x: 1/(1+np.exp(-x))),
    )
    Chinese_normalization_options = (
        ("无归一化", lambda x: x),
        ("到间隔[0，1]", lambda x: scale(x, min=0, max=1)),
        ("到间隔[-1，1]", lambda x: scale(x, min=-1, max=1)),
        ("Sigmoid函数: 1/(1+exp(-X))", lambda x: 1/(1+np.exp(-x))),
    )

    inversion_options = (
        ("No inversion", lambda x: x),
        ("-X", lambda x: -x),
        ("1 - X", lambda x: 1-x),
        ("max(X) - X", lambda x: np.max(x) - x),
        ("1/X", lambda x: 1/x),
    )
    Chinese_inversion_options = (
        ("无反转", lambda x: x),
        ("-X", lambda x: -x),
        ("1 - X", lambda x: 1-x),
        ("最大值(X) - X", lambda x: np.max(x) - x),
        ("1/X", lambda x: 1/x),
    )

    def __init__(self):
        super().__init__()

        self.data = None

        gui.radioButtons(self.controlArea, self, "normalization_method",
                         box="归一化",
                         btnLabels=[x[0] for x in self.Chinese_normalization_options],
                         callback=self._invalidate)

        gui.radioButtons(self.controlArea, self, "inversion_method",
                         box="反转",
                         btnLabels=[x[0] for x in self.Chinese_inversion_options],
                         callback=self._invalidate)

        gui.auto_apply(self.buttonsArea, self, "autocommit")

    @Inputs.distances
    def set_data(self, data):
        self.data = data
        self.commit.now()

    @gui.deferred
    def commit(self):
        distances = self.data
        if distances is not None:
            # normalize
            norm = self.normalization_options[self.normalization_method][1]
            distances = norm(distances)

            # invert
            inv = self.inversion_options[self.inversion_method][1]
            distances = inv(distances)
        self.Outputs.distances.send(distances)

    def send_report(self):
        norm, normopt = self.normalization_method, self.normalization_options
        inv, invopt = self.inversion_method, self.inversion_options
        parts = []
        if inv:
            parts.append('inversion ({})'.format(invopt[inv][0]))
        if norm:
            parts.append('normalization ({})'.format(normopt[norm][0]))
        self.report_items(
            'Model parameters',
            {'Transformation': ', '.join(parts).capitalize() or 'None'})

    def _invalidate(self):
        self.commit.deferred()


if __name__ == "__main__":  # pragma: no cover
    import Orange.distance
    data = Orange.data.Table("iris")
    dist = Orange.distance.Euclidean(data)
    WidgetPreview(OWDistanceTransformation).run(dist)
