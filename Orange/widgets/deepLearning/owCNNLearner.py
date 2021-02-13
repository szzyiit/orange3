import sys

from PyQt5.QtGui import QGuiApplication
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils.signals import Output

import torch.nn as nn
from torchsummary import summary

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
    category = 'deeplearning'

    want_main_area = True
    out_channels = [1, 2, 4, 8, 10, 16, 20, 32, 48, 64, 0]
    out_1 = 3  # 8
    out_2 = 5  # 16
    out_3 = 7  # 32
    out_4 = 5  # 16
    out_5 = 4  # 10


    class Outputs:
        model = Output('CNN 模型 (CNN model)', nn.Module, default=True, replaces=['CNN model'])

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_summary = None

        self._setup_control_area()
        gui.button(self.controlArea, self, "观察并输出模型", callback=self.check_model, autoDefault=True)

        # self.make_model()

    def _setup_control_area(self):
        settings_box = gui.widgetBox(self.controlArea, "网络设置:")
        gui.label(settings_box, self, "选择各层模型参数, 输出通道数为 0 表示此层不存在: \n")
        self.model_info = gui.label(self.mainArea, self, '模型信息')

        self.grid_type_box = gui.comboBox(
            settings_box,
            self,
            'out_1',
            items=self.out_channels[:-1],
            label='第 1 层卷积输出通道数目'
        )
        self.grid_type_box = gui.comboBox(
            settings_box,
            self,
            'out_2',
            items=self.out_channels,
            label='第 2 层卷积输出通道数目'
        )
        self.grid_type_box = gui.comboBox(
            settings_box,
            self,
            'out_3',
            items=self.out_channels,
            label='第 3 层卷积输出通道数目'
        )
        self.grid_type_box = gui.comboBox(
            settings_box,
            self,
            'out_4',
            items=self.out_channels,
            label='第 4 层卷积输出通道数目'
        )
        self.grid_type_box = gui.comboBox(
            settings_box,
            self,
            'out_5',
            items=self.out_channels,
            label='第 5 层卷积输出通道数目'
        )

    def make_model(self):
        out_channels = [self.out_channels[self.out_1],
                        self.out_channels[self.out_2],
                        self.out_channels[self.out_3],
                        self.out_channels[self.out_4],
                        self.out_channels[self.out_5],
                        ]
        out_channels = [channel for channel in out_channels if channel > 0]
        in_channels = [1] + out_channels[:-1]

        layers = [self._one_layer(in_channel, out_channel) for in_channel, out_channel in zip(in_channels[:-1], out_channels[:-1])]

        self.model = nn.Sequential(*list(layers),
                                   self.conv(in_channels[-1], out_channels[-1]),  # 1
                                   nn.BatchNorm2d(out_channels[-1]),

                                   Flatten()  # remove (1,1) grid
                                   )

    def _one_layer(self, ni, nf):
        return nn.Sequential(
            self.conv(ni, nf),
            nn.BatchNorm2d(nf),
            nn.ReLU()
        )

    def conv(self, ni, nf):
        return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)

    def check_model(self):
        self.make_model()
        sys.stdout = x = ListStream()

        print(summary(self.model, (1, 28, 28)))

        sys.stdout = sys.__stdout__

        # print(summary(self.model, (1, 28, 28)))
        s = ''.join(x.data[:-2])

        # gui.label(self.mainArea, self, s)
        # self.model_info.repaint()
        QGuiApplication.processEvents()
        self.model_info.setText(s)

        self.Outputs.model.send(self.model)


