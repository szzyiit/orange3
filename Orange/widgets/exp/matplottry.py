from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from pathlib import Path

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random



class Crawler(OWWidget):
    name = "matplot try"
    description = "训练一个爬行机器人, 熟悉强化学习的各种参数"
    # icon = "icons/robot.svg"

    want_main_area = True

    def __init__(self):
        super().__init__()
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        # self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        # self.button = QtGui.QPushButton('Plot')
        # self.button.clicked.connect(self.plot)

        # set the layout
        box = gui.vBox(self.mainArea, 'setting')
        # layout.addWidget(self.toolbar)
        box.layout().addWidget(self.canvas)
        self.plot()
        # layout.addWidget(self.button)
        # self.setLayout(layout)

    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.clear()

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()
