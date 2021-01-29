from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QFileDialog, QComboBox, QStyle, QSizePolicy
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils.signals import Output, Input
from Orange.data import Table, Domain, ContinuousVariable


class CloudScore(OWWidget):
    name = "云测评 (Cloud Scoring)"
    description = "结果上传云端进行评分"
    # icon = "icons/gridworld.png"
    category = 'evaluate'

    def __init__(self):
        super().__init__()

        gui.label(self.controlArea, self, '提交结果,自动判分,敬请期待')

    class Inputs:
        data = Input("预测结果(Predictions)", Table, replaces=['Predictions'])

    @Inputs.data
    def set_data(self):
        pass
