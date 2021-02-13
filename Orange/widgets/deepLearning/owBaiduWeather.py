from Orange.widgets.utils.owbaiduNLPApi import BaiduAPI, NLPMixin

from Orange.widgets import gui
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.settings import Setting, SettingProvider
from PyQt5.QtWidgets import QLineEdit


class WeatherReport(NLPMixin, BaiduAPI):
    name = " 天气播报(Weather Report)"
    description = "天气播报自动生成。"
    icon = "icons/rain_light.png"
    keywords = ['baidu', 'tianqi', 'bobao', 'yubao']
    category = 'deeplearning'

    URL = "https://aip.baidubce.com/rest/2.0/nlp/v1/gen_article?charset=UTF-8&access_token="
    project_id = Setting('')
    city = Setting('')

    def __init__(self):
        super().__init__()
        self.headers = {'Content-Type':'application/x-www-form-urlencoded'}
    
    def additional_controls(self):
        """
        输入本 API 所需的附加参数
        """
        settings_box = gui.widgetBox(self.controlArea, "其他参数:")
        project_id = gui.lineEdit(
            settings_box,
            self,
            "project_id",
            "输入项目 ID",
            valueType=str,
        )
        project_id.setEchoMode(QLineEdit.Password)

        city = gui.lineEdit(
            settings_box,
            self,
            "city",
            "输入城市名",
            valueType=str,
        )

    def setup_params(self):
        if self.city == '' or self.project_id == '' or self.API_KEY == '' or self.SECRET_KEY == '':
            print('error')

        self.body = {
                'project_id': self.project_id,
                'city': self.city,
            } 

    def run(self):
        self.result.setText(self.get_results()['result']['texts'][-1])
