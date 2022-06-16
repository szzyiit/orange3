from faulthandler import disable
import sys

from PyQt5.QtGui import QGuiApplication
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils.signals import Output
from Orange.widgets.settings import Setting
from AnyQt.QtWidgets import QScrollArea, QLabel
from AnyQt.QtCore import Qt

import torch.nn as nn
from torchinfo import summary
import torchvision

# https://stackoverflow.com/questions/21341096/redirect-print-to-string-list


class ListStream:
    def __init__(self):
        self.data = []

    def write(self, s):
        self.data.append(s)


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class CNNLearner(OWWidget):
    name = "卷积神经网络学习器(CNN Learner)"
    description = "构建一个训练 MNIST 数据集的简单卷积神经网络"
    icon = "icons/cnn.png"
    keywords = ['juanji', 'shenjingwangluo', 'shenduxuexi']
    category = '深度学习(DeepLearning)'

    want_main_area = True
    out_channels = [1, 2, 4, 8, 10, 16, 20, 32, 48, 64, 0]
    out_1 = 3  # 8
    out_2 = 5  # 16
    out_3 = 7  # 32
    out_4 = 5  # 16
    out_5 = 4  # 10

    class Outputs:
        model = Output('CNN 模型 (CNN model)', nn.Module,
                       default=True, replaces=['CNN model'])

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_summary = None

        self._setup_control_area()
        gui.button(self.controlArea, self, "观察并输出模型",
                   callback=self.check_model, autoDefault=True)
        

    def _setup_control_area(self):
        settings_box = gui.widgetBox(self.controlArea, "网络设置:")

        gui.label(settings_box, self, "选择各层模型参数, 输出通道数为 0 表示此层不存在: \n")
        # self.model_info = gui.label(self.mainArea, self, '模型信息')
        self.model_info = QLabel()
        self.model_info.setText('模型信息')

        self.scroll_area = QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.scroll_area.setWidget(self.model_info)
        self.scroll_area.setWidgetResizable(True)
        self.mainArea.layout().addWidget(self.scroll_area)

        grid_type_box1 = gui.comboBox(
            settings_box,
            self,
            'out_1',
            items=self.out_channels[:-1],
            label='第 1 层卷积输出通道数目'
        )

        grid_type_box2 = gui.comboBox(
            settings_box,
            self,
            'out_2',
            items=self.out_channels,
            label='第 2 层卷积输出通道数目'
        )

        grid_type_box3 = gui.comboBox(
            settings_box,
            self,
            'out_3',
            items=self.out_channels,
            label='第 3 层卷积输出通道数目'
        )

        grid_type_box4 = gui.comboBox(
            settings_box,
            self,
            'out_4',
            items=self.out_channels,
            label='第 4 层卷积输出通道数目'
        )

        grid_type_box5 = gui.comboBox(
            settings_box,
            self,
            'out_5',
            items=self.out_channels,
            label='第 5 层卷积输出通道数目'
        )
        
        self.resnet18_disabled = True
        gui.checkBox(
            self.controlArea, self, 'resnet18_disabled', '自己构建网络',
            disables=[grid_type_box1, grid_type_box2, grid_type_box3, grid_type_box4, grid_type_box5],
            tooltip="自己构建网络，否则使用预训练模型 resnet18，第一次使用预训练模型会比较慢，因为需要先下载模型."
        )


    def make_model(self):
        out_channels = [self.out_channels[self.out_1],
                        self.out_channels[self.out_2],
                        self.out_channels[self.out_3],
                        self.out_channels[self.out_4],
                        self.out_channels[self.out_5],
                        ]
        out_channels = [channel for channel in out_channels if channel > 0]
        in_channels = [3] + out_channels[:-1]

        layers = [self._one_layer(in_channel, out_channel) for in_channel, out_channel in zip(
            in_channels[:-1], out_channels[:-1])]

        if self.resnet18_disabled:
            self.model = nn.Sequential(*list(layers),
                                    self.conv(
                                        in_channels[-1], out_channels[-1]),  # 1
                                            nn.BatchNorm2d(out_channels[-1]),
                                            Flatten()  # remove (1,1) grid
                                    )
        else:
            self.model = torchvision.models.resnet18(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False

            # Parameters of newly constructed modules have requires_grad=True by default
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)

                


    def _one_layer(self, ni, nf):
        return nn.Sequential(
            self.conv(ni, nf),
            nn.BatchNorm2d(nf),
            nn.Dropout(0.25),
            nn.ReLU()
        )

    def conv(self, ni, nf):
        return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)

    def check_model(self):
        self.make_model()
        sys.stdout = x = ListStream()

        print(summary(self.model, input_size=(100, 3, 28, 28)))

        sys.stdout = sys.__stdout__

        print(summary(self.model, (100, 3, 28, 28)))
        s = ''.join(x.data[:-2])

        # gui.label(self.mainArea, self, s)
        self.model_info.repaint()
        QGuiApplication.processEvents()
        self.model_info.setText(s)

        self.Outputs.model.send(self.model)
