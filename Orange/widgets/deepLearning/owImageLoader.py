from pathlib import Path
import numpy as np

from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import QSize
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils.signals import Output

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ImageLoader(OWWidget):
    name = "图片加载器(Image Loader)"
    description = "为 MNIST 模型载入图片数据, 也可以载入其他图片. 图片会转为灰度图片"
    icon = "icons/upload.png"

    want_main_area = True
    batch_sizes = [4, 16, 64, 100, 200]
    batch_size = 3
    path_str = Setting("")

    class Outputs:
        train_data = Output('训练数据(Train Data)', DataLoader, default=True)
        test_data = Output('测试数据(Test Data)', DataLoader, default=True)

    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None
        self.images_count = 0
        # a figure instance to plot on
        self.figure = Figure()

        self.canvas = FigureCanvas(self.figure)
        self.set_size()

        box = gui.vBox(self.controlArea)
        self.path_box = gui.lineEdit(
            box,
            self,
            "path_str",
            "请输入文件夹路径",
            box="",
            valueType=str,
        )
        gui.comboBox(
            box,
            self,
            'batch_size',
            items=self.batch_sizes,
            callback=self.set_size,
            label='batch size'
        )
        gui.button(box,
                   self,
                   "载入图片",
                   callback=self.load_images,
                   )

        self.main_box = gui.vBox(self.mainArea)
        gui.label(self.main_box, self, "随机样图")
        self.main_box.layout().addWidget(self.canvas)

    def set_size(self):
        self.canvas.setMinimumSize(QSize(np.log(self.batch_sizes[self.batch_size]) * 100,
                                         np.log(self.batch_sizes[self.batch_size]) * 100))

    def load_images(self):
        path = Path(self.path_box.text())

        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5])])

        train_dataset = torchvision.datasets.ImageFolder(path/'training',
                                                         transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_sizes[self.batch_size],
                                                   shuffle=True)
        test_dataset = torchvision.datasets.ImageFolder(path/'testing',
                                                         transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=self.batch_sizes[self.batch_size],
                                                   shuffle=True)

        self.train_data = train_loader
        self.test_data = test_loader

        self.images_count = len(train_dataset) + len(test_dataset)
        self.plot()
        self.commit()

    def plot(self):
        ''' plot some random stuff '''
        def imshow(ax, img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            ax.imshow(np.transpose(npimg, (1, 2, 0)))

        dataiter = iter(self.train_data)
        images, labels = dataiter.next()

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.clear()

        imshow(ax, torchvision.utils.make_grid(images))

        # refresh canvas
        QGuiApplication.processEvents()
        self.canvas.draw()

    def commit(self):
        self.Outputs.train_data.send(self.train_data)
        self.Outputs.test_data.send(self.test_data)


