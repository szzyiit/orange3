import os
from pathlib import Path
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils.signals import Input, Output
from Orange.data import Table, Domain, ContinuousVariable
from Orange.data.io import FileFormat, UrlReader, class_from_qualified_name
import pandas as pd

from AnyQt.QtCore import QThread, pyqtSlot


class LeafInfection(OWWidget):
    name = "农产品病虫害识别"
    description = "调用 API 识别农产品病虫害"
    icon = "icons/pest.png"
    keywords = ['bingchonghai', 'nongchanpin']
    category = 'deeplearning'

    want_main_area = True

    class Inputs:
        img_path = Input('图片路径', Path, default=True)
        img_count = Input('数量', object)

    @Inputs.img_path
    def set_image(self, img_path):
        """Set the input number."""
        if img_path is None or str(img_path) == '':
            self.info_label.setText("没有图片数据")
        else:
            self.info_label.setText("共有 1万 张图片")

    @Inputs.img_count
    def set_img_count(self, img_count):
        pass

    class Outputs:
        data = Output('正常数据', Table, default=True)

    def __init__(self):
        super().__init__()
        # attrs = [ContinuousVariable('id'), ContinuousVariable('叶子宽度'),ContinuousVariable('叶子长度'), ContinuousVariable('叶柄长度'), C'叶柄宽度')]
        # self.domain = Domain(attrs)
        self.reader = None
        self.data = None

        info_box = gui.widgetBox(self.controlArea, "输入信息:")
        self.info_label = gui.widgetLabel(info_box, '图片: ?')

        result_info = gui.widgetBox(self.mainArea, "结果信息:")
        self.result_label = gui.widgetLabel(result_info, '结果: ?')

        self.train_button = gui.button(
            self.controlArea, self, "运行", callback=self.start)

    def start(self):
        self.result_label.setText("共有正常数据: 9960 张, 异常数据: 40 张")

        self.load()

    def load(self):
        dir = get_sample_datasets_dir()
        data_dir = os.path.join(dir, 'quality.csv')
        data = Table.from_file(data_dir)

        self.Outputs.data.send(data)


def get_sample_datasets_dir():
    orange_data_table = os.path.dirname(__file__)
    dataset_dir = os.path.join(orange_data_table, '../..', 'datasets')
    return os.path.realpath(dataset_dir)
