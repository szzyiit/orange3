from pathlib import Path
import numpy as np
import os

from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QFileDialog, QComboBox, QStyle, QSizePolicy
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils import stdpaths
from Orange.widgets.utils.signals import Output

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg


class ImageLoader(OWWidget):
    name = "图片加载器(Image Loader)"
    description = "为 MNIST 模型载入图片数据, 也可以载入其他图片. 图片会转为灰度图片"
    icon = "icons/upload.png"
    keywords = ['tupian', 'tuxiang', 'zairu', 'zairutupian', 'jiazai', 'jiazaitupian', 'daoru']
    category = 'deeplearning'

    want_main_area = True
    batch_sizes = [4, 16, 64, 100, 200]
    batch_size = 3
    for_train_or_not = [True, False]
    for_train = 0
    # path_str = Setting("")
    #: List of recent filenames.
    history = Setting([])
    #: Current (last selected) filename or None.
    dir_name = Setting(None)

    class Outputs:
        train_data = Output('训练数据(Train Data)', DataLoader, default=True, replaces=['Data'])
        test_data = Output('测试数据(Test Data)', DataLoader, default=True, replaces=['Data'])
        image = Output('图片(Image)', Path, replaces=['Image', 'Path'])

    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None
        self.selectedIndex = -1
        self.dataiter = None
        self.images = []
        self.image_index = 0
        self.image_count = 0
        # a figure instance to plot on
        self.figure = Figure()

        self.canvas = FigureCanvas(self.figure)
        self.set_size()

        self._set_control_area()

        self.main_box = gui.vBox(self.mainArea)
        self.main_label = gui.label(self.main_box, self, "随机样图")
        self.main_box.layout().addWidget(self.canvas)

    def _set_control_area(self):
        box = gui.widgetBox(self.controlArea, '设置:')
        box.layout().setAlignment(Qt.AlignTop)

        dir_box = gui.widgetBox(box, '图片文件夹:',  orientation=Qt.Horizontal)
        self.filesCB = gui.comboBox(
            dir_box, self, "selectedIndex", callback=self._on_recent)
        self.filesCB.setMinimumContentsLength(20)
        self.filesCB.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLength)

        self.loadbutton = gui.button(dir_box, self, "...", callback=self.browse)
        self.loadbutton.setIcon(
            self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.loadbutton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)


        # filter valid existing filenames
        self.history = list(filter(os.path.isdir, self.history))[:20]
        for dir_name in self.history:
            self.filesCB.addItem(os.path.basename(dir_name), userData=dir_name)

        # restore the current selection if the filename is
        # in the history list
        if self.dir_name in self.history:
            self.selectedIndex = self.history.index(self.dir_name)
        else:
            self.selectedIndex = -1
            self.dir_name = None


        gui.radioButtonsInBox(box, self, 'for_train',
                              btnLabels=['需要训练', '只要预测'],
                              label='是否需要训练',
                              orientation=Qt.Horizontal,
                              callback=self.update_batch_setting)

        gui.separator(box, height=10)
        self.batch_box = gui.comboBox(
            box,
            self,
            'batch_size',
            items=self.batch_sizes,
            callback=self.set_size,
            label='batch size'
        )

        gui.separator(box, height=10)
        gui.button(box,
                   self,
                   "载入图片",
                   callback=self.load_images,
                   )
        self.next_image_button = gui.button(box,
                   self,
                   "下一组图片",
                   callback=self.next_image,
                   )
        self.next_image_button.setEnabled(False)

    def _on_recent(self):
        self.load(self.history[self.selectedIndex])

    def load(self, dir_name):
        """Load the object from filename and send it to output."""
        self._remember(dir_name)

    def _remember(self, dir_name):
        """
        Remember `filename` was accessed.
        """
        if dir_name in self.history:
            index = self.history.index(dir_name)
            del self.history[index]
            self.filesCB.removeItem(index)

        self.history.insert(0, dir_name)

        self.filesCB.insertItem(0, os.path.basename(dir_name),
                                userData=dir_name)
        self.selectedIndex = 0
        self.dir_name = dir_name


    def browse(self):
        """Select a filename using an open file dialog."""
        if self.dir_name is None:
            startdir = stdpaths.Documents
        else:
            startdir = os.path.dirname(self.dir_name)

        self.dir_name = QFileDialog.getExistingDirectory(
            self,
            "Open a folder",
            startdir,
            QFileDialog.ShowDirsOnly
        )
        self.load(self.dir_name)

    def set_size(self):
        self.canvas.setMinimumSize(QSize(np.log(self.batch_sizes[self.batch_size]) * 100,
                                         np.log(self.batch_sizes[self.batch_size]) * 100))

    def update_batch_setting(self):
        self.batch_box.setEnabled(self.for_train == 0)
        self.next_image_button.setEnabled(self.for_train == 1)
        self.image_index = 0

    def next_image(self):
        self.image_index = self.image_index % (self.image_count - 1) + 1
        self.plot(self.images)

    def load_images(self):
        # a simple custom collate function, just to show the idea
        def my_collate(batch):
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            target = torch.LongTensor(target)
            return [data, target]

        path = Path(self.dir_name)

        try:
            if self.for_train_or_not[self.for_train]:
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

                self.dataiter = iter(self.train_data)
                images, labels = self.dataiter.next()
            else:
                self.images = images = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.bmp'))
                self.image_count = len(images)

        except:
            # TODO: a bug when right/wrong, need to find out
            QGuiApplication.processEvents()
            self.main_label.setText('图片载入出错')
            self.plot(None, clear=True)
        else:
            self.plot(images)
            self.commit()

    def plot(self, images, clear=False):
        ''' plot some random stuff '''
        def imshow(ax, img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            ax.imshow(np.transpose(npimg, (1, 2, 0)))

        def imshow_one(ax, img):
            image = mpimg.imread(img)
            ax.imshow(image)
            self.Outputs.image.send(img)


        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.clear()

        if clear:
            self.canvas.draw()
            return

        if self.for_train == 0:
            imshow(ax, torchvision.utils.make_grid(images))
        else:
            if self.image_count != 0:
                imshow_one(ax, images[self.image_index])

        # refresh canvas
        QGuiApplication.processEvents()
        self.canvas.draw()
        self.main_label.setText('随机样图')

    def commit(self):
        self.Outputs.train_data.send(self.train_data)
        self.Outputs.test_data.send(self.test_data)


