from pathlib import Path
import numpy as np
import os
from PyQt5.QtCore import QProcess

from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QFileDialog, QComboBox, QStyle, QSizePolicy
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils import stdpaths

from PIL import Image



class ImageConverter(OWWidget):
    name = "PNG2WebP"
    description = "png to webp"


    def __init__(self):
        super().__init__()
        self.input_path = ''
        self.output_path = ''

        self._set_control_area()


    def _set_control_area(self):
        box = gui.widgetBox(self.controlArea, '文件夹位置:')

        gui.lineEdit( box, self, "input_path", "输入文件夹位置", valueType=str,)

        gui.lineEdit( box, self, "output_path", "输出文件夹位置", valueType=str,)

        self.commit_button = gui.button(
            self.controlArea, self, "运行", callback=self.run, autoDefault=True)

    def run(self):
        for root, dirs, files in os.walk(self.input_path):
            for filename in files:
                print(filename)
                if filename.endswith('.png'):
                    im = Image.open(f'{root}/{filename}')

                    if im.size[0] > 2000:
                        out_width = 1200
                    elif im.size[0] > 1200:
                        out_width = min(int(im.size[0] / 3 * 2), 1200)
                    else:
                        out_width = im.size[0]
                    command = f'cwebp -resize {out_width} 0 {root}/{filename} -o {root}/out/{filename}.webp'

                    self.process = QProcess(self)

                    self.process.started.connect(self.onstart)
                    self.process.finished.connect(self.onfinish)
                    self.process.start(command)

    def onstart(self):
        self.commit_button.setEnabled(False)

    def onfinish(self):
        self.commit_button.setEnabled(True)




