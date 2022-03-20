import base64
import time
import requests
import json
import urllib.parse
import hashlib
import random
import string
from pathlib import Path
from PyQt5.QtGui import QGuiApplication

from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem, QLineEdit
from PyQt5.QtGui import QStandardItemModel

from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets import gui
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.settings import Setting, SettingProvider
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable

from torch.utils.data import DataLoader


class ImageMixin:
    """
    添加腾讯人工智能图像类功能
    """
    class Inputs:
        img_path = Input('图片路径', Path, default=True)

    @Inputs.img_path
    def set_image(self, img_path):
        """Set the input number."""
        if img_path is None or str(img_path) == '':
            self.info_label.setText("没有图片数据")
        else:
            with open(img_path, 'rb') as imageFile:
                self.image_str = base64.b64encode(imageFile.read())


class TencentAPI(OWWidget):
    """
    使用腾讯人工智能 API 做深度学习预测
    """

    want_main_area = True
    URL = ''
    APPID = Setting('')
    APPKEY = Setting('')
    row_count = 10
    column_count = 2
    params = None

    #
    # class Outputs:
    #     data = Output('预测结果', Table, default=True)

    def __init__(self):
        super().__init__()
        self.time_stamp = 0
        self.response = {}

        self.info_box = gui.widgetBox(self.controlArea, "数据信息")
        self.info_label = gui.label(self.info_box, self, '使用腾讯 AI 开放平台')

        self._setup_control_area()
        self.additional_control()

        gui.button(self.controlArea, self, "运行", callback=self.run, autoDefault=True)

        self._setup_main_area()
        # self.response_label = gui.label(self.mainArea, self, '')

    def _setup_main_area(self):
        self.tableWidget = QTableWidget()
        # set row count
        self.tableWidget.setRowCount(self.row_count)

        # set column count
        self.tableWidget.setColumnCount(self.column_count)
        # self.tableWidget.setItem(0, 0, QTableWidgetItem("Cell (1,1)"))

        self.mainArea.layout().addWidget(self.tableWidget)

    def _setup_control_area(self):
        settings_box = gui.widgetBox(self.controlArea, "参数设置:")
        appid = gui.lineEdit(
            settings_box,
            self,
            "APPID",
            "输入 APPID",
            valueType=str,
        )
        appid.setEchoMode(QLineEdit.Password)
        appkey = gui.lineEdit(
            settings_box,
            self,
            "APPKEY",
            "输入 APPKEY",
            valueType=str,
        )
        appkey.setEchoMode(QLineEdit.Password)

    def additional_control(self):
        pass

    def get_results(self, res):
        pass

    def show_results(self, res):
        if isinstance(res, list):
            keys = [key for key in res[0]]
            self.tableWidget.setHorizontalHeaderLabels(keys)
            self.column_count = len(keys)

            QGuiApplication.processEvents()
            self.tableWidget.setRowCount(0)
            self.tableWidget.setRowCount(self.row_count)
            for index, item in enumerate(res):
                cols = [item[key] for key in keys]
                for i, value in enumerate(cols):
                    self.tableWidget.setItem(index, i, QTableWidgetItem(str(value)))
        else:
            print(res)

    def show_errors(self, error):
        self.info_label.setText('服务器或网络不稳定,识别失败')
        print(error)
        # self.response_label.setText(error)

    def update_results(self):
        pass

    def setup_params(self):
        pass

    def run(self):
        # self.load_data()
        # with open('/Users/sziit/Desktop/maxresdefault.jpg', 'rb') as imageFile:
        #     image_str = base64.b64encode(imageFile.read())
        nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 10))
        self.params = {'app_id': self.APPID, 'time_stamp': int(time.time()),
                  'nonce_str': nonce_str}
        self.setup_params()
        sign = self.get_sign_code(self.params, self.APPKEY)
        self.params['sign'] = sign
        rt = requests.post(self.URL, data=self.params)
        jsonData = json.loads(rt.text)
        if jsonData.get('ret') == 0:
            self.response = jsonData.get('data')
            r = self.get_results(self.response)
            self.show_results(r)
        else:
            self.show_errors(jsonData)

    def get_sign_code(self, params, app_key):
        ''' 生成签名CODE

        1. 计算步骤
        用于计算签名的参数在不同接口之间会有差异，但算法过程固定如下4个步骤。
        将<key, value>请求参数对按key进行字典升序排序，得到有序的参数对列表N
        将列表N中的参数对按URL键值对的格式拼接成字符串，得到字符串T（如：key1=value1&key2=value2），URL键值拼接过程value部分需要URL编码，URL编码算法用大写字母，例如%E8，而不是小写%e8
        将应用密钥以app_key为键名，组成URL键值拼接到字符串T末尾，得到字符串S（如：key1=value1&key2=value2&app_key=密钥)
        对字符串S进行MD5运算，将得到的MD5值所有字符转换成大写，得到接口请求签名
        2. 注意事项
        不同接口要求的参数对不一样，计算签名使用的参数对也不一样
        参数名区分大小写，参数值为空不参与签名
        URL键值拼接过程value部分需要URL编码
        签名有效期5分钟，需要请求接口时刻实时计算签名信息
        :param params: 参数字典
        :param app_key:
        :return:
        '''
        if params is None or type(params) != dict or len(params) == 0: return
        try:
            params = sorted(params.items(), key=lambda x: x[0])
            _str = ''
            for item in params:
                key = item[0]
                value = item[1]
                if value == '': continue
                _str += urllib.parse.urlencode({key: value}) + '&'
            _str += 'app_key=' + app_key
            _str = hashlib.md5(_str.encode('utf-8')).hexdigest()
            return _str.upper()
        except Exception as e:
            self.info_label.setText('服务器或网络不稳定,识别失败')
            print(f'sign code error: ${e}')
