"""Tree learner widget"""

from collections import OrderedDict

from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.modelling.tree import TreeLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWTreeLearner(OWBaseLearner):
    """Tree algorithm with forward pruning."""
    name = "树(Tree)"
    description = "一种前向剪枝的树算法"
    icon = "icons/Tree.svg"
    replaces = [
        "Orange.widgets.classify.owclassificationtree.OWClassificationTree",
        "Orange.widgets.regression.owregressiontree.OWRegressionTree",
        "Orange.widgets.classify.owclassificationtree.OWTreeLearner",
        "Orange.widgets.regression.owregressiontree.OWTreeLearner",
    ]
    priority = 30
    keywords = ["Classification Tree", 'jueceshu', 'fenleishu', 'shu', 'huiguishu']
    category = 'model'

    LEARNER = TreeLearner

    binary_trees = Setting(True)
    limit_min_leaf = Setting(True)
    min_leaf = Setting(2)
    limit_min_internal = Setting(True)
    min_internal = Setting(5)
    limit_depth = Setting(True)
    max_depth = Setting(100)

    # Classification only settings
    limit_majority = Setting(True)
    sufficient_majority = Setting(95)

    spin_boxes = (
        ("叶中的最小实例数: ",
         "limit_min_leaf", "min_leaf", 1, 1000),
        ("不要拆分小于以下值的子集: ",
         "limit_min_internal", "min_internal", 1, 1000),
        ("将树最大深度限制为: ",
         "limit_depth", "max_depth", 1, 1000))

    classification_spin_boxes = (
        ("当多数达到 [%] 时停止: ",
         "limit_majority", "sufficient_majority", 51, 100),)

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, '参数')
        # the checkbox is put into vBox for alignemnt with other checkboxes
        gui.checkBox(gui.vBox(box), self, "binary_trees", "归纳二叉树",
                     callback=self.settings_changed)
        for label, check, setting, fromv, tov in self.spin_boxes:
            gui.spin(box, self, setting, fromv, tov, label=label,
                     checked=check, alignment=Qt.AlignRight,
                     callback=self.settings_changed,
                     checkCallback=self.settings_changed, controlWidth=80)

    def add_classification_layout(self, box):
        for label, check, setting, minv, maxv in self.classification_spin_boxes:
            gui.spin(box, self, setting, minv, maxv,
                     label=label, checked=check, alignment=Qt.AlignRight,
                     callback=self.settings_changed, controlWidth=80,
                     checkCallback=self.settings_changed)

    def learner_kwargs(self):
        # Pylint doesn't get our Settings
        # pylint: disable=invalid-sequence-index
        return dict(
            max_depth=(None, self.max_depth)[self.limit_depth],
            min_samples_split=(2, self.min_internal)[self.limit_min_internal],
            min_samples_leaf=(1, self.min_leaf)[self.limit_min_leaf],
            binarize=self.binary_trees,
            preprocessors=self.preprocessors,
            sufficient_majority=(1, self.sufficient_majority / 100)[
                self.limit_majority])

    def create_learner(self):
        # pylint: disable=not-callable
        return self.LEARNER(**self.learner_kwargs())

    def get_learner_parameters(self):
        from Orange.widgets.report import plural_w
        items = OrderedDict()
        items["Pruning"] = ", ".join(s for s, c in (
            (plural_w("at least {number} instance{s} in leaves",
                      self.min_leaf), self.limit_min_leaf),
            (plural_w("at least {number} instance{s} in internal nodes",
                      self.min_internal), self.limit_min_internal),
            ("maximum depth {}".format(self.max_depth), self.limit_depth)
        ) if c) or "None"
        if self.limit_majority:
            items["Splitting"] = "Stop splitting when majority reaches %d%% " \
                                 "(classification only)" % \
                                 self.sufficient_majority
        items["Binary trees"] = ("No", "Yes")[self.binary_trees]
        return items


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWTreeLearner).run(Table("iris"))
