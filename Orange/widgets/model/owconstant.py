from Orange.data import Table
from Orange.modelling.constant import ConstantLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWConstant(OWBaseLearner):
    name = "常量预测(Constant)"
    description = "从训练集预测最频繁的类或平均值。"
    icon = "icons/Constant.svg"
    replaces = [
        "Orange.widgets.classify.owmajority.OWMajority",
        "Orange.widgets.regression.owmean.OWMean",
    ]
    priority = 10
    keywords = ["majority", "mean", 'changshu', 'changliang']

    LEARNER = ConstantLearner
    category = 'model'


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWConstant).run(Table("iris"))
