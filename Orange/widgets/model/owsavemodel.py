import pickle

from Orange.widgets.widget import Input
from Orange.base import Model
from Orange.widgets.utils.save.owsavebase import OWSaveBase
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWSaveModel(OWSaveBase):
    name = "保存模型(Save Model)"
    description = "将经过训练的模型保存到输出文件。"
    icon = "icons/SaveModel.svg"
    replaces = ["Orange.widgets.classify.owsaveclassifier.OWSaveClassifier"]
    priority = 3000
    keywords = ['moxing', 'baocunmoxing']
    category = 'model'

    class Inputs:
        model = Input("模型(Model)", Model, replaces=['Model'])

    filters = ["Pickled model (*.pkcls)"]

    @Inputs.model
    def set_model(self, model):
        self.data = model
        self.on_new_input()

    def do_save(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self.data, f)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSaveModel).run()
