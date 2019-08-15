import numpy

import Orange.data
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets import gui
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import QDockWidget, QFrame, QHBoxLayout, QPushButton


class OWDataSamplerA(OWWidget):
    name = "Plotly experiment"
    description = "Randomly selects a subset of instances from the dataset"
    icon = "icons/DataSamplerA.svg"
    priority = 10

    def __init__(self):
        super().__init__()

        # # GUI
        my_web = QWebEngineView()

        self.layout().addWidget(my_web)
        my_web.load(QUrl("https://www.bilibili.com/"))

