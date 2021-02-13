from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from pathlib import Path
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import QScrollArea, QLabel
import sys


class GridWold(OWWidget):
    name = "格子世界(Grid World)"
    description = "在格子世界中熟悉不确定的世界"
    icon = "icons/gridworld.png"
    manual_mode, auto_mode = 0, 1
    mode = manual_mode
    keywords = ['gezishijie']
    category = 'reinforcement'

    discount = 0.9
    noise_ratio = 0.2
    living_reward = 0
    epsilon = 0.3
    # learning_rate = 0.5
    iterations = 10
    episodes = 1
    grids = ['BookGrid', 'BridgeGrid', 'CliffGrid', 'MazeGrid']
    grid_type = 0
    agents = ['random', 'value', 'q']
    agent_type = 0

    dir_path = Path(__file__).resolve()
    parent_path = dir_path.parent.parent
    if sys.platform.startswith('win'):
        command = f'{str(parent_path)}/binaries/gridworld.exe '
    else:
        command = f'{str(parent_path)}/binaries/gridworld '
    command_options = f'-m -n {noise_ratio} -g {grid_type}'

    want_main_area = True

    def __init__(self):
        super().__init__()
        self.add_main_layout()
        self.final_command = ''

    def add_main_layout(self):
        self._add_common_settings_box()
        self._add_mode_selection_box()
        self._add_settings_box()
        self._commit_button()
        self._add_main_area()



    def _add_common_settings_box(self):
        settings_box = gui.widgetBox(self.controlArea, "通用设置:")

        self.grid_type_box = gui.comboBox(
            settings_box,
            self,
            'grid_type',
            items=self.grids,
            label='世界类型'
        )

    def _add_mode_selection_box(self):
        mode_box = gui.hBox(self.controlArea, "模式选择")

        gui.radioButtonsInBox(
            mode_box,
            self,
            "mode",
            btnLabels=['手动模式', '自动模式'],
        )

    def _add_settings_box(self):
        settings_box = gui.vBox(self.controlArea, "更多设置")

        self.noise_ratio_spin = gui.doubleSpin(
            settings_box,
            self,
            "noise_ratio",
            minv=0,
            maxv=1,
            step=0.1,
            label="噪音比例:",
        )

        self.discount_spin = gui.doubleSpin(
            settings_box,
            self,
            "discount",
            minv=0.1,
            maxv=0.9,
            step=0.1,
            label='折扣比例')

        self.living_reward_spin = gui.doubleSpin(
            settings_box,
            self,
            'living_reward',
            minv=-2,
            maxv=1,
            step=0.1,
            label='生存回报:'
        )

        self.epsilon_spin = gui.doubleSpin(
            settings_box,
            self,
            'epsilon',
            minv=0,
            maxv=1,
            step=0.1,
            label='贪婪程度epsilon',
        )

        self.iterations_slider = gui.hSlider(
            settings_box,
            self,
            "iterations",
            minValue=0,
            maxValue=100,
            label='迭代次数',
        )

        self.episodes_spin = gui.hSlider(
            settings_box,
            self,
            "episodes",
            minValue=0,
            maxValue=20,
            label='尝试次数',
        )

        self.agent_type_box = gui.comboBox(
            settings_box,
            self,
            'agent_type',
            items=self.agents,
            label='智能体类型'
        )

        # self._update_enable_settings(False)

    def _commit_button(self):
        self.commit_button = gui.button(self.controlArea, self, "运行", callback=self.commit,
                                        toggleButton=True, autoDefault=True)

    def _add_main_area(self):
        self.output_label = QLabel()
        self.scroll_area = QScrollArea()

        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.output_label)
        self.scroll_area.setWidgetResizable(True)
        self.mainArea.layout().addWidget(self.scroll_area)

    # def _update_mode(self):
    #     if self.mode == self.manual_mode:
    #         self._update_enable_settings(False)
    #     else:
    #         self._update_enable_settings(True)

    # def _update_enable_settings(self, auto_mode=True):
    #     self.noise_ratio_spin.setEnabled(True)
    #     self.discount_spin.setEnabled(auto_mode)
    #     self.living_reward_spin.setEnabled(auto_mode)
    #     self.epsilon_spin.setEnabled(auto_mode)
    #     self.iterations_slider.setEnabled(auto_mode)
    #     self.agent_type_box.setEnabled(auto_mode)

    def commit(self):
        # dir_path = Path.cwd()
        # dir_path = Path(__file__).resolve()
        # parent_path = dir_path.parent.parent
        self.process = QProcess(self)

        # self.p = subprocess.Popen([f'{dir_path}/Orange/widgets/binaries/gridworld', '-m', f'-n {self.proportion/100}'],
        #                cwd=f'{dir_path}')
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.construct_command(self.mode)
        self.process.started.connect(self.onstart)
        self.process.finished.connect(self.onfinish)
        self.process.start(self.final_command)

    def construct_command(self, mode):
        options = f'-v -d {self.discount} -r {self.living_reward} -e {self.epsilon} ' \
                  f'-i {self.iterations} -g {self.grids[self.grid_type]} ' \
                  f'-a {self.agents[self.agent_type]} -n {self.noise_ratio} ' \
                  f'-k {self.episodes}'
        if mode == self.manual_mode:
            self.command_options = '-m ' + options
        else:
            self.command_options = options

        self.final_command = self.command + self.command_options
        print(self.final_command)

    def onstart(self):
        self.commit_button.setEnabled(False)

    def onfinish(self):
        self.commit_button.setEnabled(True)

    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput(), encoding='utf-8')
        # print(text)
        self.output_label.setText(text)

