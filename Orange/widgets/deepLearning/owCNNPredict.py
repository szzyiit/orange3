import sys
from pathlib import Path
import base64
import numpy as np

from PyQt5.QtGui import QGuiApplication
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils.signals import Input
from PyQt5.QtWidgets import QLabel

import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from PIL import Image


class CNNPredict(OWWidget):
    name = "卷积神经网络预测(CNN  Predict)"
    description = "使用训练好的 MNIST 模型预测"
    # icon source: https://dryicons.com/icon/crystal-ball-icon-11458
    icon = "icons/predicts.svg"
    keywords = ['yuce', 'juanji', 'shenjingwangluo', 'shenduxuexi']
    category = 'deeplearning'

    want_main_area = False

    class Inputs:
        img_path = Input('图片路径', Path, default=True)
        model = Input('模型(Model)', nn.Module, replaces=['Model'])

    @Inputs.img_path
    def set_image(self, img_path):
        """Set the input number."""
        if img_path is None or str(img_path) == '':
            # self.info_label.setText("没有图片数据")
            self.data_ready = False
        else:
            self.data_ready = True
            self.img_path = img_path

    @Inputs.model
    def set_model(self, model):
        """Set the input number."""
        if model is None:
            # self.info_label_model.setText("必须有模型")
            self.model_ready = False
        else:
            self.model = model
            # self.predict(self.img_path)
            self.model_ready = True

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_summary = None
        self.model_ready = False
        self.data_ready = False

        self._setup_control_area()

    def handleNewSignals(self):
        if self.model_ready and self.data_ready:
            self.predict()

    def _setup_control_area(self):
        self.info_label = QLabel()   
        self.controlArea.layout().addWidget(self.info_label)


    def image_loader(self, loader):
        """load image, returns cuda tensor"""
        image = Image.open(self.img_path)
        image = loader(image).float()
        image = image.unsqueeze(0) 
        return image

    def predict(self):
        transform = transforms.Compose(
            [transforms.Resize(28),
                transforms.Grayscale(num_output_channels=1), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])

        image = self.image_loader(transform)

        output = self.model(image)

        prediction = int(torch.max(output.data, 1)[1].numpy())

        number = 3 if prediction == 0 else 7

        self.info_label.setText(f'这个图片数字是 {number}')
