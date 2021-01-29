"""Naive Bayes Learner
"""

from Orange.data import Table
from Orange.classification.naive_bayes import NaiveBayesLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWNaiveBayes(OWBaseLearner):
    name = "朴素贝叶斯(Naive Bayes)"
    description = "特征独立假设基础上, 基于贝叶斯定理的的快速简单的概率分类器"
    icon = "icons/NaiveBayes.svg"
    replaces = [
        "Orange.widgets.classify.ownaivebayes.OWNaiveBayes",
    ]
    priority = 70
    keywords = ['pusubeiyesi', 'beiyesi']
    category = 'model'

    LEARNER = NaiveBayesLearner


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWNaiveBayes).run(Table("iris"))
