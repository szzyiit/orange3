from Orange.widgets.widget import OWWidget, Input
from Orange.widgets import gui
from pathlib import Path
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import QScrollArea


class GridWold(OWWidget):
    name = "Print"
    description = "Print out a number"
    icon = "icons/print.svg"
    proportion = 50

    # class Inputs:
    #     number = Input("Number", int)

    want_main_area = True

    def __init__(self):
        super().__init__()
        self.box = gui.widgetBox(self.controlArea, "设置噪音比例:")
        gui.spin(
            self.box,
            self,
            "proportion",
            minv=10,
            maxv=90,
            step=10,
            label="噪音比例:",
        )

        self.commit_button = gui.button(self.box, self, "运行", callback=self.commit,
                                        toggleButton=True, autoDefault=True)

        self.output_label = gui.widgetLabel(self.mainArea, "sdfsfd")
        self.output_label.setWordWrap(True)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.output_label)


    def commit(self):
        # dir_path = Path.cwd()
        dir_path = Path(__file__).resolve()
        parent_path = dir_path.parent.parent
        self.process = QProcess(self)

        # self.p = subprocess.Popen([f'{dir_path}/Orange/widgets/binaries/gridworld', '-m', f'-n {self.proportion/100}'],
        #                cwd=f'{dir_path}')
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.start(f'{str(parent_path)}/binaries/gridworld -m -n {self.proportion/100}')
        self.process.started.connect(self.onstart)
        self.process.finished.connect(self.onfinish)

    def onstart(self):
        self.commit_button.setEnabled(False)

    def onfinish(self):
        self.commit_button.setEnabled(True)

    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput(), encoding='utf-8')

        self.output_label.setText(text)

