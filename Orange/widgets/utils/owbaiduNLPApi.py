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


class NLPMixin(OWWidget):
    """
    添加百度自然语言处理功能
    """

    want_main_area = True

    API_KEY = Setting('')
    SECRET_KEY = Setting('')

    params = None

    #
    # class Outputs:
    #     data = Output('预测结果', Table, default=True)

    def __init__(self):
        super().__init__()

        self.response = {}

        self.info_box = gui.widgetBox(self.controlArea, "信息")
        self.info_label = gui.label(self.info_box, self, '使用百度自然语言处理平台')

        self._setup_control_area()

        gui.button(self.controlArea, self, "运行", callback=self.run, autoDefault=True)

        self._setup_main_area()
        # self.response_label = gui.label(self.mainArea, self, '')

    def _setup_main_area(self):

        self.result = gui.label(self.mainArea, self, '')
        self.result.setWordWrap(True)




    def _setup_control_area(self):
        settings_box = gui.widgetBox(self.controlArea, "秘钥设置:")
        appid = gui.lineEdit(
            settings_box,
            self,
            "API_KEY",
            "输入 API_KEY",
            valueType=str,
        )
        appid.setEchoMode(QLineEdit.Password)
        appkey = gui.lineEdit(
            settings_box,
            self,
            "SECRET_KEY",
            "输入 SECRET_KEY",
            valueType=str,
        )
        appkey.setEchoMode(QLineEdit.Password)

        self.additional_controls()

    def additional_controls(self):
        """
        设置每个应用不同的 UI 组件
        """
        pass
    
    def run(self):
        raise NotImplementedError


class BaiduAPI():
    """
    使用百度自然语言处理 API 做深度学习预测
    """
    URL = ''

    def __init__(self, params):
        self.headers = {}
        self.body = {}


    def show_errors(self, error):
        self.info_label.setText('服务器或网络不稳定,识别失败')
        print(error)
        # self.response_label.setText(error)



    def setup_params(self):
        raise NotImplementedError

    def get_results(self):
        access_token = self.get_access_token()
        r = requests.post(self.URL + access_token, headers=self.headers, data=self.body)

        rtext = r.text
        return json.loads(rtext)

    def get_access_token(self):
        self.setup_params()
        host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.API_KEY}&client_secret={self.SECRET_KEY}'

        try:
            response = requests.post(host)
            json_data = response.json()
            access_token = json_data["access_token"]
        except:
            print('get access_token error')
            access_token = None

        return access_token



