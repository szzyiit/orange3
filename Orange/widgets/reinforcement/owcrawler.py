from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from pathlib import Path
from PyQt5.QtCore import QProcess
import sys


class Crawler(OWWidget):
    name = "爬行者(Crawler)"
    description = "训练一个爬行机器人, 熟悉强化学习的各种参数"
    icon = "icons/robot.svg"
    keywords = ['paixingzhe']
    category = 'reinforcement'

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.commit_button = gui.button(self.controlArea, self, "运行", callback=self.commit,
                                        toggleButton=True, autoDefault=True)

    def commit(self):
        dir_path = Path(__file__).resolve()
        parent_path = dir_path.parent.parent
        if sys.platform.startswith('win'):
            command = f'{str(parent_path)}/binaries/crawler.exe'
        else:
            command = f'{str(parent_path)}/binaries/crawler'

        self.process = QProcess(self)

        self.process.started.connect(self.onstart)
        self.process.finished.connect(self.onfinish)
        self.process.start(command)


    def onstart(self):
        self.commit_button.setEnabled(False)

    def onfinish(self):
        self.commit_button.setEnabled(True)
