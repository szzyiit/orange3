from Orange.widgets.widget import OWWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from pathlib import Path


class OWPlayground(OWWidget):
    name = "TensorFlow 游乐场(TensorFlow Playground)"
    description = "使用TensorFlow 游乐场自由尝试深度学习"
    icon = "icons/playground.png"
    priority = 10
    keywords = ['youlechang', 'shenduxuexi']
    category = 'deeplearning'

    resizing_enabled = False

    def __init__(self):
        super().__init__()

        # # GUI
        my_web = QWebEngineView()

        dir_path = Path(__file__).resolve().parent
        file = dir_path/'playground/index.html'

        self.layout().addWidget(my_web)
        my_web.resize(1100, 700)
        my_web.load(QUrl().fromLocalFile(str(file)))

