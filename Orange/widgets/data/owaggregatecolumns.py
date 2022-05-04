from typing import List

import numpy as np

from AnyQt.QtWidgets import QSizePolicy
from AnyQt.QtCore import Qt
from Orange.data import Variable, Table, ContinuousVariable, TimeVariable
from Orange.data.util import get_unique_names
from Orange.widgets import gui, widget
from Orange.widgets.settings import (
    ContextSetting, Setting, DomainContextHandler
)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils.itemmodels import DomainModel


class OWAggregateColumns(widget.OWWidget):
    name = "聚合列(Aggregate Columns)"
    description = "计算选定列的和，最大值，最小值等."
    category = "数据(Data)"
    icon = "icons/AggregateColumns.svg"
    priority = 100
    keywords = ['juhe', "aggregate", "sum", "product", "max", "min", "mean",
                "median", "variance"]

    class Inputs:
        data = Input("数据(Data)", Table, default=True, replaces=['Data'])

    class Outputs:
        data = Output("数据(Data)", Table, replaces=['Data'])

    want_main_area = False

    settingsHandler = DomainContextHandler()
    variables: List[Variable] = ContextSetting([])
    operation = Setting("和")
    var_name = Setting("agg")
    auto_apply = Setting(True)

    Operations = {"和": np.nansum, "积": np.nanprod,
                  "最小": np.nanmin, "最大": np.nanmax,
                  "平均": np.nanmean, "方差": np.nanvar,
                  "中位数": np.nanmedian}
    TimePreserving = ("最小", "最大", "平均", "中位数")

    def __init__(self):
        super().__init__()
        self.data = None

        box = gui.vBox(self.controlArea, box=True)

        self.variable_model = DomainModel(
            order=DomainModel.MIXED, valid_types=(ContinuousVariable, ))
        var_list = gui.listView(
            box, self, "variables", model=self.variable_model,
            callback=self.commit.deferred
        )
        var_list.setSelectionMode(var_list.ExtendedSelection)

        combo = gui.comboBox(
            box, self, "operation",
            label="操作: ", orientation=Qt.Horizontal,
            items=list(self.Operations), sendSelectedValue=True,
            callback=self.commit.deferred
        )
        combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        gui.lineEdit(
            box, self, "var_name",
            label="变量名: ", orientation=Qt.Horizontal,
            callback=self.commit.deferred
        )

        gui.auto_apply(self.controlArea, self)

    @Inputs.data
    def set_data(self, data: Table = None):
        self.closeContext()
        self.variables.clear()
        self.data = data
        if self.data:
            self.variable_model.set_domain(data.domain)
            self.openContext(data)
        else:
            self.variable_model.set_domain(None)
        self.commit.now()

    @gui.deferred
    def commit(self):
        augmented = self._compute_data()
        self.Outputs.data.send(augmented)

    def _compute_data(self):
        if not self.data or not self.variables:
            return self.data

        new_col = self._compute_column()
        new_var = self._new_var()
        return self.data.add_column(new_var, new_col)

    def _compute_column(self):
        arr = np.empty((len(self.data), len(self.variables)))
        for i, var in enumerate(self.variables):
            arr[:, i] = self.data.get_column_view(var)[0].astype(float)
        func = self.Operations[self.operation]
        return func(arr, axis=1)

    def _new_var_name(self):
        return get_unique_names(self.data.domain, self.var_name)

    def _new_var(self):
        name = self._new_var_name()
        if self.operation in self.TimePreserving \
                and all(isinstance(var, TimeVariable) for var in self.variables):
            return TimeVariable(name)
        return ContinuousVariable(name)

    def send_report(self):
        # fp for self.variables, pylint: disable=unsubscriptable-object
        if not self.data or not self.variables:
            return
        var_list = ", ".join(f"'{var.name}'"
                             for var in self.variables[:31][:-1])
        if len(self.variables) > 30:
            var_list += f" and {len(self.variables) - 30} others"
        else:
            var_list += f" and '{self.variables[-1].name}'"
        self.report_items((
            ("Output:",
             f"'{self._new_var_name()}' as {self.operation.lower()} of {var_list}"
             ),
        ))


if __name__ == "__main__":  # pragma: no cover
    brown = Table("brown-selected")
    WidgetPreview(OWAggregateColumns).run(set_data=brown)
