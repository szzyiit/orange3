from AnyQt.QtWidgets import QFrame

from Orange.data import Table
from Orange.preprocess.remove import Remove
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output


class OWPurgeDomain(widget.OWWidget):
    name = "清除特征(Purge Domain)"
    description = "从数据集中删除冗余值和特征,对值进行排序。"
    icon = "icons/PurgeDomain.svg"
    category = "数据(Data)"
    keywords = ["remove", "delete", "unused", 'qinli', 'qinchu']

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
            parent.layout().addWidget(frame)

        boxAt = gui.vBox(self.controlArea, "特征")
        for value, label in self.feature_options:
            gui.checkBox(boxAt, self, value, label,
                         callback=self.commit.deferred)
        add_line(boxAt)
        gui.label(boxAt, self,
                  "已排序: %(resortedAttrs)s, "
                  "已减少: %(reducedAttrs)s, 已删除: %(removedAttrs)s")

        boxAt = gui.vBox(self.controlArea, "类别")
        for value, label in self.class_options:
            gui.checkBox(boxAt, self, value, label,
                         callback=self.commit.deferred)
        add_line(boxAt)
        gui.label(boxAt, self,
                  "已排序: %(resortedClasses)s,"
                  "已减少: %(reducedClasses)s, 已删除: %(removedClasses)s")

        boxAt = gui.vBox(self.controlArea, "元属性")
        for value, label in self.meta_options:
            gui.checkBox(boxAt, self, value, label,
                         callback=self.commit.deferred)
        add_line(boxAt)
        gui.label(boxAt, self,
                  "已减少: %(reducedMetas)s, 已删除: %(removedMetas)s")

        gui.auto_send(self.buttonsArea, self, "autoSend")

    @Inputs.data
    @check_sql_input
    def setData(self, dataset):
        if dataset is not None:
            self.data = dataset
            self.commit.now()
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

    @gui.deferred
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
