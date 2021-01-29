from Orange.classification import CalibratedLearner, ThresholdLearner, \
    NaiveBayesLearner
from Orange.data import Table
from Orange.modelling import Learner
from Orange.widgets import gui
from Orange.widgets.widget import Input
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWCalibratedLearner(OWBaseLearner):
    name = "校准器(Calibrated Learner)"
    description = "在其他学习器之外包裹概率校准和决策阈值优化功能"
    icon = "icons/CalibratedLearner.svg"
    priority = 20
    keywords = ["calibration", "threshold", 'jiaozhunqi', 'jiaodui']
    category = 'model'

    LEARNER = CalibratedLearner

    SigmoidCalibration, IsotonicCalibration, NoCalibration = range(3)
    CalibrationOptions = ("Sigmoid 校准",
                          "Isotonic 校准",
                          "不校准")
    CalibrationShort = ("Sigmoid", "Isotonic", "")
    CalibrationMap = {
        SigmoidCalibration: CalibratedLearner.Sigmoid,
        IsotonicCalibration: CalibratedLearner.Isotonic}

    OptimizeCA, OptimizeF1, NoThresholdOptimization = range(3)
    ThresholdOptions = ("优化分类准确率",
                        "优化 F1 score",
                        "无阈值优化")
    ThresholdShort = ("CA", "F1", "")
    ThresholdMap = {
        OptimizeCA: ThresholdLearner.OptimizeCA,
        OptimizeF1: ThresholdLearner.OptimizeF1}

    learner_name = Setting("", schema_only=True)
    calibration = Setting(SigmoidCalibration)
    threshold = Setting(OptimizeCA)

    class Inputs(OWBaseLearner.Inputs):
        base_learner = Input("基学习器(Base Learner)", Learner, replaces=['Base Learner'])

    def __init__(self):
        super().__init__()
        self.base_learner = None

    def add_main_layout(self):
        gui.radioButtons(
            self.controlArea, self, "calibration", self.CalibrationOptions,
            box="概率校准",
            callback=self.calibration_options_changed)
        gui.radioButtons(
            self.controlArea, self, "threshold", self.ThresholdOptions,
            box="决策阈值优化",
            callback=self.calibration_options_changed)

    @Inputs.base_learner
    def set_learner(self, learner):
        self.base_learner = learner
        self._set_default_name()
        self.unconditional_apply()

    def _set_default_name(self):
        if self.base_learner is None:
            self.name = "Calibrated learner"
        else:
            self.name = " + ".join(part for part in (
                self.base_learner.name.title(),
                self.CalibrationShort[self.calibration],
                self.ThresholdShort[self.threshold]) if part)
        self.controls.learner_name.setPlaceholderText(self.name)

    def calibration_options_changed(self):
        self._set_default_name()
        self.apply()

    def create_learner(self):
        class IdentityWrapper(Learner):
            def fit_storage(self, data):
                return self.base_learner.fit_storage(data)

        if self.base_learner is None:
            return None
        learner = self.base_learner
        if self.calibration != self.NoCalibration:
            learner = CalibratedLearner(learner,
                                        self.CalibrationMap[self.calibration])
        if self.threshold != self.NoThresholdOptimization:
            learner = ThresholdLearner(learner,
                                       self.ThresholdMap[self.threshold])
        if self.preprocessors:
            if learner is self.base_learner:
                learner = IdentityWrapper()
            learner.preprocessors = (self.preprocessors, )
        return learner

    def get_learner_parameters(self):
        return (("Calibrate probabilities",
                 self.CalibrationOptions[self.calibration]),
                ("Threshold optimization",
                 self.ThresholdOptions[self.threshold]))


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCalibratedLearner).run(
        Table("heart_disease"),
        set_learner=NaiveBayesLearner())
