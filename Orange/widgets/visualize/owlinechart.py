from numbers import Number
from collections import OrderedDict
from os.path import join, dirname

import numpy as np

from Orange.data import TimeVariable, Table
from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Input

from AnyQt.QtWidgets import QTreeWidget, \
    QWidget, QPushButton, QListView, QVBoxLayout
from AnyQt.QtGui import QIcon
from AnyQt.QtCore import QSize, pyqtSignal, QTimer

from Orange.widgets.utils.itemmodels import VariableListModel
# from orangecontrib.timeseries import Timeseries
from .utils.highcharts import Highchart


class PlotConfigWidget(QWidget, gui.OWComponent):
    sigClosed = pyqtSignal(str, QWidget)
    sigLogarithmic = pyqtSignal(str, bool)
    sigType = pyqtSignal(str, str)
    sigSelection = pyqtSignal(str, list)

    is_logarithmic = False
    plot_types = {'线':'line', 
                    '折线':'step line', 
                    '柱':'column', 
                    '面积':'area', 
                    '样条曲线':'spline'}
    plot_type = '线'

    def __init__(self, owwidget, ax, varmodel):
        QWidget.__init__(self, owwidget)
        gui.OWComponent.__init__(self)

        self.ax = ax
        self.view = view = QListView(self, selectionMode=QTreeWidget.ExtendedSelection,)
        view.setModel(varmodel)
        selection = view.selectionModel()
        selection.selectionChanged.connect(self.selection_changed)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self.setLayout(box)

        hbox = gui.hBox(self)
        gui.comboBox(hbox, self, 'plot_type',
                     label='类型:',
                     orientation='horizontal',
                     items=('线', '折线', '柱', '面积', '样条曲线'),
                     sendSelectedValue=True,
                     callback=lambda: self.sigType.emit(ax, self.plot_types[self.plot_type]))
        gui.rubber(hbox)
        self.button_close = button = QPushButton('×', hbox,
                                                 visible=False,
                                                 minimumSize=QSize(20, 20),
                                                 maximumSize=QSize(20, 20),
                                                 styleSheet='''
                                                     QPushButton {
                                                         font-weight: bold;
                                                         font-size:14pt;
                                                         margin:0;
                                                         padding:0;
                                                     }''')
        button.clicked.connect(lambda: self.sigClosed.emit(ax, self))
        hbox.layout().addWidget(button)
        gui.checkBox(self, self, 'is_logarithmic', '对数轴',
                     callback=lambda: self.sigLogarithmic.emit(ax, self.is_logarithmic))
        box.addWidget(view)

    # This is here because sometimes enterEvent/leaveEvent below were called
    # before the constructor set button_close appropriately. I have no idea.
    button_close = None

    def enterEvent(self, event):
        if self.button_close:
            self.button_close.setVisible(True)

    def leaveEvent(self, event):
        if self.button_close:
            self.button_close.setVisible(False)

    def selection_changed(self):
        selection = [mi.model()[mi.row()]
                     for mi in self.view.selectionModel().selectedIndexes()]
        self.sigSelection.emit(self.ax, selection)
        self.sigType.emit(self.ax, self.plot_types[self.plot_type])


class Highstock(Highchart):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args,
                         yAxis_lineWidth=2,
                         yAxis_labels_x=6,
                         yAxis_labels_y=-3,
                         yAxis_labels_align_x='right',
                         yAxis_title_text=None,
                         enable_scrollbar = False,
                         plotOptions_series_dataGrouping_groupPixelWidth=2,
                         plotOptions_series_dataGrouping_approximation='high',
                         plotOptions_areasplinerange_states_hover_lineWidthPlus=0,
                         plotOptions_areasplinerange_tooltip_pointFormat='''
                            <span style="color:{point.color}">\u25CF</span>
                            {series.name}: <b>{point.low:.2f} – {point.high:.2f}</b><br/>''',
                         **kwargs)
        self.parent = parent
        self.axes = []

    def _resizeAxes(self):
        if not self.axes:
            return
        MARGIN = 2
        HEIGHT = (100 - (len(self.axes) - 1) * MARGIN) // len(self.axes)
        self.evalJS('''
            var SKIP_AXES = 1,
                HEIGHT = %(HEIGHT)f,
                MARGIN = %(MARGIN)f;
            for (var i = 0; i < chart.yAxis.length - SKIP_AXES; ++i) {
                var top_offset = i * (HEIGHT + MARGIN);
                chart.yAxis[i + SKIP_AXES].update({
                    top: top_offset + '%%',
                    height: HEIGHT + '%%',
                    offset: 0  // Fixes https://github.com/highcharts/highcharts/issues/5199
                }, false);
            }
            chart.reflow();
            chart.redraw(false);
        ''' % locals())

    def addAxis(self):
        from random import random
        ax = 'ax_' + str(random())[2:]
        self.axes.append(ax)
        self.evalJS('''
            chart.addAxis({
                id: '%(ax)s',
            }, false, false, false);
        ''' % locals())
        self._resizeAxes()
        # TODO: multiple series on the bottom navigator, http://jsfiddle.net/highcharts/SD4XN/
        return ax

    def add_legend(self):
        self.evalJS('''chart.legend.options.enabled = true;''')

    def removeAxis(self, ax):
        self.axes.remove(ax)
        self.evalJS('''
            chart.get('%(ax)s').remove();
        ''' % dict(ax=ax))
        self._resizeAxes()

    def setXAxisType(self, ax_type):
        self.evalJS('''
        for (var i=0; i<chart.xAxis.length; ++i) {
            chart.xAxis[i].update({type: '%s'});
        }
        ''' % ax_type)

    def setSeries(self, ax, series):
        """TODO: Clean this shit up"""
        newseries = []
        names = []
        deltas = []
        ci_percents = []

        data = self.parent.data

        for attr in series:
            newseries.append(data.get_column_view(attr)[0])
            names.append(attr.name)
            deltas.append(None)
            ci_percents.append(None)

        self.exposeObject('series_' + ax, {'data': newseries,
                                           'ci_percents': ci_percents,
                                           'names': names,
                                           'deltas': deltas})
        self.evalJS('''
            var ax = chart.get('%(ax)s');
            chart.series
            .filter(function(s) { return s.yAxis == ax })
            .map(function(s) { s.remove(false); });

            var data = series_%(ax)s.data,
                names = series_%(ax)s.names,
                deltas = series_%(ax)s.deltas,
                ci_percents = series_%(ax)s.ci_percents;

            for (var i=0; i < data.length; ++i) {
                var opts = {
                    data: data[i],
                    name: names[i],
                    yAxis: '%(ax)s'
                };

                if (deltas[i]) {
                    opts.pointStart = deltas[i][0];
                    // skip 1: pointEnd (forecast start)
                    opts.pointInterval = deltas[i][2];
                    if (deltas[i][3])
                        opts.pointIntervalUnit = deltas[i][3];
                }

                var added_series = chart.addSeries(opts, false, false);
                
            }
            chart.redraw(false);
        ''' % dict(ax=ax))

    def setLogarithmic(self, ax, is_logarithmic):
        self.evalJS('''
            chart.get('%(ax)s').update({ type: '%(type)s', allowNegativeLog:
            %(negative)s, tickInterval: %(tick)s,
        });
        ''' % dict(ax=ax, type='logarithmic' if is_logarithmic else 'linear',
                   negative='true' if is_logarithmic else 'false',
                   tick=1 if is_logarithmic else 'undefined'))
        if not is_logarithmic:
            # it is a workaround for Highcharts issue - Highcharts do not
            # un-mark data as null when changing graph from logarithmic to
            # linear
            self.evalJS(
                '''
                s = chart.get('%(ax)s').series;
                s.forEach(function(series) {
                    series.data.forEach(function(point) {
                        point.update({'isNull': false});
                    });
                });
                ''' % dict(ax=ax)
            )

    def setType(self, ax, type):
        step, type = ('true', 'line') if type == 'step line' else ('false', type)
        self.evalJS('''
            var ax = chart.get('%(ax)s');
            chart.series
            .filter(function(s) { return s.yAxis == ax; })
            .map(function(s) {
                s.update({
                    type: '%(type)s',
                    step: %(step)s
                }, false);
            });
            chart.redraw(false);
        ''' % locals())


class OWLineChart(widget.OWWidget):
    name = '折线图(Line Chart)'
    description = "以折线图形式观察某一列数据."
    icon = 'icons/LineChart.svg'
    priority = 90
    category = 'visualize'
    keywords = ['zhexiantu']

    class Inputs:
        time_series = Input("数据(Data)", Table, replaces='Data')

    attrs = settings.Setting({})  # Maps data.name -> [attrs]

    graph_name = 'chart'

    def __init__(self):
        self.data = None
        self.plots = []
        self.configs = []
        self.varmodel = VariableListModel(parent=self)
        icon = QIcon(join(dirname(__file__), 'icons', 'LineChart-plus.png'))
        self.add_button = button = QPushButton(icon, '添加折线图', self)
        button.clicked.connect(self.add_plot)
        self.controlArea.layout().addWidget(button)
        self.configsArea = gui.vBox(self.controlArea)
        self.controlArea.layout().addStretch(1)
        # TODO: allow selecting ranges that are sent to output as subset table
        self.chart = highstock = Highstock(self, highchart='StockChart')
        self.mainArea.layout().addWidget(highstock)
        # highstock.evalJS('Highcharts.setOptions({navigator: {enabled:false}});')
        highstock.chart(
            # For some reason, these options don't work as global opts applied at Highstock init time
            # Disable top range selector
            rangeSelector_enabled=False,
            rangeSelector_inputEnabled=False,
            # Disable bottom miniview navigator (it doesn't update)
            navigator_enabled=False, )
        QTimer.singleShot(0, self.add_plot)
        self.chart.add_legend()

    def add_plot(self):
        ax = self.chart.addAxis()
        config = PlotConfigWidget(self, ax, self.varmodel)
        # Connect the signals
        config.sigSelection.connect(self.chart.setSeries)
        config.sigLogarithmic.connect(self.chart.setLogarithmic)
        config.sigType.connect(self.chart.setType)
        config.sigClosed.connect(self.chart.removeAxis)
        config.sigClosed.connect(lambda ax, widget: widget.setParent(None))
        config.sigClosed.connect(lambda ax, widget:
                                 self.add_button.setDisabled(False))
        self.configs.append(config)
        self.add_button.setDisabled(len(self.configs) >= 5)
        config.sigClosed.connect(lambda ax, widget: self.remove_plot(widget))

        self.configsArea.layout().addWidget(config)

    def remove_plot(self, plot):
        self.configs.remove(plot)
        # # self.configsArea.layout()
        if len(self.chart.axes) < 2:
            self.resize(QSize(925, 635))

    @Inputs.time_series
    def set_data(self, data):
        # TODO: set xAxis resolution and tooltip time contents depending on
        # data.time_delta. See: http://imgur.com/yrnlgQz

        # If the same data is updated, short circuit to just updating the chart,
        # retaining all panels and list view selections ...

        # new_data = None if data is None else \
        #            Timeseries.from_data_table(data)
        new_data = data
        if new_data is not None and self.data is not None \
                and new_data.domain == self.data.domain:
            self.data = new_data
            for config in self.configs:
                config.selection_changed()
            return

        # self.data = data = None if data is None else \
        #                    Timeseries.from_data_table(data)
        self.data = data
        if data is None:
            self.varmodel.clear()
            self.chart.clear()
            return
        # if getattr(data.time_variable, 'utc_offset', False):
        #     offset_minutes = data.time_variable.utc_offset.total_seconds() / 60
        #     self.chart.evalJS('Highcharts.setOptions({global: {timezoneOffset: %d}});' % -offset_minutes)  # Why is this negative? It works.
        #     self.chart.chart() and

        self.chart.setXAxisType(
            'linear')

        self.varmodel.wrap([var for var in data.domain.variables
                            if var.is_continuous ])



