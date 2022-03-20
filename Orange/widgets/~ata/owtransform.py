from typing import Optional

from Orange.data import Table, Domain
from Orange.widgets import gui
from Orange.widgets.report.report import describe_data
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg


class OWTransform(OWWidget):
    name = "变换(Apply Domain)"
    description = "在数据表上应用模板域"
    icon = "icons/Transform.svg"
    priority = 2110
    keywords = ["transform"]

    class Inputs:
        data = Input("数据(Data)", Table, default=True, replaces=['Data'])
        template_data = Input("模板数据(Template Data)", Table, replaces=['Template Data'])

    class Outputs:
        transformed_data = Output("转换的数据(Transformed Data)", Table, replaces=['Transformed Data'])

    class Error(OWWidget.Error):
        error = Msg("An error occurred while transforming data.\n{}")

    resizing_enabled = False
    want_main_area = False

    def __init__(self):
        super().__init__()
        self.data = None  # type: Optional[Table]
        self.template_domain = None  # type: Optional[Domain]
        self.transformed_info = describe_data(None)  # type: OrderedDict

        info_box = gui.widgetBox(self.controlArea, "信息")
        self.input_label = gui.widgetLabel(info_box, "")
        self.template_label = gui.widgetLabel(info_box, "")
        self.output_label = gui.widgetLabel(info_box, "")
        self.set_input_label_text()
        self.set_template_label_text()

    def set_input_label_text(self):
        text = "没有输入数据。"
        if self.data:
            text = "输入包含 {:,} 个实例 {:,} 个特征的数据。".format(
                len(self.data),
                len(self.data.domain.attributes))
        self.input_label.setText(text)

    def set_template_label_text(self):
        text = "没有输入模板数据。"
        if self.data and self.template_domain is not None:
            text = "模板域已应用。"
        elif self.template_domain is not None:
            text = "模板数据包括 {:,} 个特征。".format(
                len(self.template_domain.attributes))
        self.template_label.setText(text)

    def set_output_label_text(self, data):
        text = ""
        if data:
            text = "输出数据包括 {:,} 个特征。".format(
                len(data.domain.attributes))
        self.output_label.setText(text)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.data = data
        self.set_input_label_text()

    @Inputs.template_data
    @check_sql_input
    def set_template_data(self, data):
        self.template_domain = data and data.domain

    def handleNewSignals(self):
        self.apply()

    def apply(self):
        self.clear_messages()
        transformed_data = None
        if self.data and self.template_domain is not None:
            try:
                transformed_data = self.data.transform(self.template_domain)
            except Exception as ex:  # pylint: disable=broad-except
                self.Error.error(ex)

        data = transformed_data
        self.transformed_info = describe_data(data)
        self.Outputs.transformed_data.send(data)
        self.set_template_label_text()
        self.set_output_label_text(data)

    def send_report(self):
        if self.data:
            self.report_data("Data", self.data)
        if self.template_domain is not None:
            self.report_domain("Template data", self.template_domain)
        if self.transformed_info:
            self.report_items("Transformed data", self.transformed_info)


if __name__ == "__main__":  # pragma: no cover
    from Orange.preprocess import Discretize

    table = Table("iris")
    WidgetPreview(OWTransform).run(
        set_data=table, set_template_data=Discretize()(table))
