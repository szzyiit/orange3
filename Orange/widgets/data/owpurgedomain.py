from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFrame

from Orange.data import Table
from Orange.preprocess.remove import Remove
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.widget import Input, Output


class OWPurgeDomain(widget.OWWidget):
    name = "清除特征(Purge Domain)"
    description = "从数据集中删除冗余值和特征,对值进行排序。"
    icon = "icons/PurgeDomain.svg"
    category = "Data"
    keywords = ["remove", "delete", "unused", 'qinli', 'qinchu', 'tezheng']

    class Inputs:
        data = Input("数据(Data)", Table, replaces=['Data'])

    class Outputs:
        data = Output("数据(Data)", Table, replaces=['Data'])

    removeValues = Setting(1)
    removeAttributes = Setting(1)
    removeClasses = Setting(1)
    removeClassAttribute = Setting(1)
    removeMetaAttributeValues = Setting(1)
    removeMetaAttributes = Setting(1)
    autoSend = Setting(True)
    sortValues = Setting(True)
    sortClasses = Setting(True)

    want_main_area = False
    resizing_enabled = False
    buttons_area_orientation = Qt.Vertical

    feature_options = (('sortValues', '分类特征值排序'),
                       ('removeValues', '删除未使用的特征值'),
                       ('removeAttributes', '删除常数特征'))

    class_options = (('sortClasses', '分类值排序'),
                     ('removeClasses', '删除未使用的类别'),
                     ('removeClassAttribute', '移除常数类别'))

    meta_options = (('removeMetaAttributeValues', '删除未使用的元属性值'),
                    ('removeMetaAttributes', '移除常数元属性'))

    stat_labels = (('已排序的特征', 'resortedAttrs'),
                   ('减少的特征', 'reducedAttrs'),
                   ('删除的特征', 'removedAttrs'),
                   ('已排序的类别', 'resortedClasses'),
                   ('减少的分类', 'reducedClasses'),
                   ('删除的类别', 'removedClasses'),
                   ('减少的元属性', 'reducedMetas'),
                   ('删除的元属性', 'removedMetas'))

    def __init__(self):
        super().__init__()
        self.data = None

        self.removedAttrs = "-"
        self.reducedAttrs = "-"
        self.resortedAttrs = "-"
        self.removedClasses = "-"
        self.reducedClasses = "-"
        self.resortedClasses = "-"
        self.removedMetas = "-"
        self.reducedMetas = "-"

        def add_line(parent):
            frame = QFrame()
            frame.setFrameShape(QFrame.HLine)
            frame.setFrameShadow(QFrame.Sunken)
            parent.layout().addSpacing(6)
            parent.layout().addWidget(frame)
            parent.layout().addSpacing(6)

        boxAt = gui.vBox(self.controlArea, "特征")
        for not_first, (value, label) in enumerate(self.feature_options):
            if not_first:
                gui.separator(boxAt, 2)
            gui.checkBox(boxAt, self, value, label,
                         callback=self.optionsChanged)
        add_line(boxAt)
        gui.label(boxAt, self,
                  "排序了: %(resortedAttrs)s, "
                  "减少了: %(reducedAttrs)s, 删除了: %(removedAttrs)s")

        boxAt = gui.vBox(self.controlArea, "类别(Classes)", addSpace=True)
        for not_first, (value, label) in enumerate(self.class_options):
            if not_first:
                gui.separator(boxAt, 2)
            gui.checkBox(boxAt, self, value, label,
                         callback=self.optionsChanged)
        add_line(boxAt)
        gui.label(boxAt, self,
                  "排序了: %(resortedClasses)s,"
                  "减少了: %(reducedClasses)s, 删除了: %(removedClasses)s")

        boxAt = gui.vBox(self.controlArea, "元属性", addSpace=True)
        for not_first, (value, label) in enumerate(self.meta_options):
            if not_first:
                gui.separator(boxAt, 2)
            gui.checkBox(boxAt, self, value, label,
                         callback=self.optionsChanged)
        add_line(boxAt)
        gui.label(boxAt, self,
                  "减少了: %(reducedMetas)s, 删除了: %(removedMetas)s")

        gui.auto_send(self.buttonsArea, self, "autoSend")
        gui.rubber(self.controlArea)

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

    @Inputs.data
    @check_sql_input
    def setData(self, dataset):
        if dataset is not None:
            self.data = dataset
            self.info.set_input_summary(len(dataset),
                                        format_summary_details(dataset))
            self.unconditional_commit()
        else:
            self.removedAttrs = "-"
            self.reducedAttrs = "-"
            self.resortedAttrs = "-"
            self.removedClasses = "-"
            self.reducedClasses = "-"
            self.resortedClasses = "-"
            self.removedMetas = "-"
            self.reducedMetas = "-"
            self.Outputs.data.send(None)
            self.data = None
            self.info.set_input_summary(self.info.NoInput)
            self.info.set_output_summary(self.info.NoOutput)

    def optionsChanged(self):
        self.commit()

    def commit(self):
        if self.data is None:
            return

        attr_flags = sum([Remove.SortValues * self.sortValues,
                          Remove.RemoveConstant * self.removeAttributes,
                          Remove.RemoveUnusedValues * self.removeValues])
        class_flags = sum([Remove.SortValues * self.sortClasses,
                           Remove.RemoveConstant * self.removeClassAttribute,
                           Remove.RemoveUnusedValues * self.removeClasses])
        meta_flags = sum([Remove.RemoveConstant * self.removeMetaAttributes,
                          Remove.RemoveUnusedValues * self.removeMetaAttributeValues])
        remover = Remove(attr_flags, class_flags, meta_flags)
        cleaned = remover(self.data)
        attr_res, class_res, meta_res = \
            remover.attr_results, remover.class_results, remover.meta_results

        self.removedAttrs = attr_res['removed']
        self.reducedAttrs = attr_res['reduced']
        self.resortedAttrs = attr_res['sorted']

        self.removedClasses = class_res['removed']
        self.reducedClasses = class_res['reduced']
        self.resortedClasses = class_res['sorted']

        self.removedMetas = meta_res['removed']
        self.reducedMetas = meta_res['reduced']

        self.info.set_output_summary(len(cleaned),
                                     format_summary_details(cleaned))
        self.Outputs.data.send(cleaned)

    def send_report(self):
        def list_opts(opts):
            return "; ".join(label.lower()
                             for value, label in opts
                             if getattr(self, value)) or "no changes"

        self.report_items("Settings", (
            ("Features", list_opts(self.feature_options)),
            ("Classes", list_opts(self.class_options)),
            ("Metas", list_opts(self.meta_options))))
        if self.data:
            self.report_items("Statistics", (
                (label, getattr(self, value))
                for label, value in self.stat_labels
            ))


if __name__ == "__main__":  # pragma: no cover
    data = Table.from_url("https://datasets.biolab.si/core/car.tab")
    subset = [inst for inst in data if inst["buying"] == "v-high"]
    subset = Table(data.domain, subset)
    # The "buying" should be removed and the class "y" reduced
    WidgetPreview(OWPurgeDomain).run(subset)
