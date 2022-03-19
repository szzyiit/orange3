from functools import reduce
from types import SimpleNamespace

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QGridLayout

import Orange.data
from Orange.util import Reprable
from Orange.statistics import distribution
from Orange.preprocess import Continuize
from Orange.preprocess.transformation import Identity, Indicator, Normalizer
from Orange.data.table import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.widget import Input, Output


class OWContinuize(widget.OWWidget):
    name = "连续化(Continuize)"
    description = ("将分类属性转换为数值属性，还可以归一化这些值" )
    icon = "icons/Continuize.svg"
    category = 'Data'
    keywords = ['lianxu', 'shuzhi']

    class Inputs:
        data = Input("数据(Data)", Orange.data.Table, replaces=['Data'])

    class Outputs:
        data = Output("数据(Data)", Orange.data.Table, replaces=['Data'])

    want_main_area = False
    buttons_area_orientation = Qt.Vertical
    resizing_enabled = False

    Normalize = SimpleNamespace(Leave=0, Standardize=1, Center=2, Scale=3,
                                Normalize11=4, Normalize01=5)

    settings_version = 2
    multinomial_treatment = Setting(0)
    continuous_treatment = Setting(Normalize.Leave)
    class_treatment = Setting(0)
    autosend = Setting(True)

    multinomial_treats = (
        ("以第一个值为基数", Continuize.FirstAsBase),
        ("最常见值作为基数", Continuize.FrequentAsBase),
        ("每个值一个属性", Continuize.Indicators),
        ("忽略多值属性", Continuize.RemoveMultinomial),
        ("删除分类属性", Continuize.Remove),
        ("按有序数据处理", Continuize.AsOrdinal),
        ("除以取值数目", Continuize.AsNormalizedOrdinal))

    continuous_treats = (
        ("保持原样", True),
        ("标准归一化为 μ=0, σ²=1", False),
        ("中心设置到 μ=0", False),
        ("缩放到 σ²=1", True),
        ("归一化到 [-1, 1]", False),
        ("归一化到 [0, 1]", False)
    )

    class_treats = (
        ("保持原样", Continuize.Leave),
        ("按有序数据处理", Continuize.AsOrdinal),
        ("除以取值数目", Continuize.AsNormalizedOrdinal),
        ("每个值一个分类", Continuize.Indicators),
    )

    def __init__(self):
        super().__init__()

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, orientation=layout)

        box = gui.radioButtonsInBox(
            None, self, "multinomial_treatment", box="分类特征",
            btnLabels=[x[0] for x in self.multinomial_treats],
            callback=self.settings_changed)
        gui.rubber(box)
        layout.addWidget(box, 0, 0, 2, 1)

        box = gui.radioButtonsInBox(
            None, self, "continuous_treatment", box = "数值特征",
            btnLabels=[x[0] for x in self.continuous_treats],
            callback=self.settings_changed)
        box.layout().addStretch(10)
        layout.addWidget(box, 0, 1, 2, 1)

        box = gui.radioButtonsInBox(
            None, self, "class_treatment", box="分类结果",
            btnLabels=[t[0] for t in self.class_treats],
            callback=self.settings_changed)
        box.layout().addStretch(10)
        layout.addWidget(box, 0, 2)

        ac = gui.auto_apply(None, self, "autosend", box=False)
        layout.addWidget(ac, 1, 2)

        self.data = None
        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

    def settings_changed(self):
        self.commit()

    @Inputs.data
    @check_sql_input
    def setData(self, data):
        self.data = data
        self.enable_normalization()
        if data is None:
            self.info.set_input_summary(self.info.NoInput)
            self.info.set_output_summary(self.info.NoOutput)
            self.Outputs.data.send(None)
        else:
            self.info.set_input_summary(len(data),
                                        format_summary_details(data))
            self.unconditional_commit()

    def enable_normalization(self):
        buttons = self.controls.continuous_treatment.buttons
        if self.data is not None and self.data.is_sparse():
            if self.continuous_treatment == self.Normalize.Standardize:
                self.continuous_treatment = self.Normalize.Scale
            else:
                self.continuous_treatment = self.Normalize.Leave
            for button, (_, supports_sparse) \
                    in zip(buttons, self.continuous_treats):
                button.setEnabled(supports_sparse)
        else:
            for button in buttons:
                button.setEnabled(True)

    def constructContinuizer(self):
        conzer = DomainContinuizer(
            multinomial_treatment=self.multinomial_treats[self.multinomial_treatment][1],
            continuous_treatment=self.continuous_treatment,
            class_treatment=self.class_treats[self.class_treatment][1]
        )
        return conzer

    def commit(self):
        continuizer = self.constructContinuizer()
        if self.data:
            domain = continuizer(self.data)
            data = self.data.transform(domain)
            self.Outputs.data.send(data)
            self.info.set_output_summary(len(data),
                                         format_summary_details(data))
        else:
            self.Outputs.data.send(self.data)  # None or empty data

    def send_report(self):
        self.report_items(
            "Settings",
            [("Categorical features",
              self.multinomial_treats[self.multinomial_treatment][0]),
             ("Numeric features",
              self.continuous_treats[self.continuous_treatment][0]),
             ("Class", self.class_treats[self.class_treatment][0])])

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            Normalize = cls.Normalize
            cont_treat = settings.pop("continuous_treatment", 0)
            zero_based = settings.pop("zero_based", True)
            if cont_treat == 1:
                if zero_based:
                    settings["continuous_treatment"] = Normalize.Normalize01
                else:
                    settings["continuous_treatment"] = Normalize.Normalize11
            elif cont_treat == 2:
                settings["continuous_treatment"] = Normalize.Standardize


class WeightedIndicator(Indicator):
    def __init__(self, variable, value, weight=1.0):
        super().__init__(variable, value)
        self.weight = weight

    def transform(self, c):
        t = super().transform(c) * self.weight
        if self.weight != 1.0:
            t *= self.weight
        return t

    def __eq__(self, other):
        return super().__eq__(other) and self.weight == other.weight

    def __hash__(self):
        return hash((type(self), self.variable, self.value, self.weight))


def make_indicator_var(source, value_ind, weight=None):
    if weight is None:
        indicator = Indicator(source, value=value_ind)
    else:
        indicator = WeightedIndicator(source, value=value_ind, weight=weight)
    return Orange.data.ContinuousVariable(
        "{}={}".format(source.name, source.values[value_ind]),
        compute_value=indicator
    )


def dummy_coding(var, base_value=0):
    N = len(var.values)
    return [make_indicator_var(var, i)
            for i in range(N) if i != base_value]


def one_hot_coding(var):
    N = len(var.values)
    return [make_indicator_var(var, i) for i in range(N)]


def continuize_domain(data,
                      multinomial_treatment=Continuize.Indicators,
                      continuous_treatment=OWContinuize.Normalize.Leave,
                      class_treatment=Continuize.Leave):
    domain = data.domain
    def needs_dist(var, mtreat, ctreat):
        "Does the `var` need a distribution given specified flags"
        if var.is_discrete:
            return mtreat == Continuize.FrequentAsBase
        elif var.is_continuous:
            return ctreat != OWContinuize.Normalize.Leave
        else:
            raise ValueError

    # Compute the column indices which need a distribution.
    attr_needs_dist = [needs_dist(var, multinomial_treatment,
                                  continuous_treatment)
                       for var in domain.attributes]
    cls_needs_dist = [needs_dist(var, class_treatment, OWContinuize.Normalize.Leave)
                      for var in domain.class_vars]

    columns = [i for i, needs in enumerate(attr_needs_dist + cls_needs_dist)
               if needs]

    if columns:
        if data is None:
            raise TypeError("continuizer requires data")
        dist = distribution.get_distributions_for_columns(data, columns)
    else:
        dist = []

    dist_iter = iter(dist)

    newattrs = [continuize_var(var, next(dist_iter) if needs_dist else None,
                               multinomial_treatment, continuous_treatment)
                for var, needs_dist in zip(domain.attributes, attr_needs_dist)]
    newclass = [continuize_var(var,
                               next(dist_iter) if needs_dist else None,
                               class_treatment, OWContinuize.Normalize.Leave)
                for var, needs_dist in zip(domain.class_vars, cls_needs_dist)]

    newattrs = reduce(list.__iadd__, newattrs, [])
    newclass = reduce(list.__iadd__, newclass, [])
    return Orange.data.Domain(newattrs, newclass, domain.metas)


def continuize_var(var,
                   data_or_dist=None,
                   multinomial_treatment=Continuize.Indicators,
                   continuous_treatment=OWContinuize.Normalize.Leave):
    def continuize_continuous():
        dist = _ensure_dist(var, data_or_dist) if continuous_treatment != OWContinuize.Normalize.Leave else None
        treatments = [lambda var, _: var,
                      normalize_by_sd, center_to_mean, divide_by_sd,
                      normalize_to_11, normalize_to_01]
        if dist is not None and dist.shape[1] == 0:
            return [var]
        new_var = treatments[continuous_treatment](var, dist)
        return [new_var]

    def continuize_discrete():
        if len(var.values) > 2 and \
                multinomial_treatment == Continuize.ReportError:
            raise ValueError("{0.name} is a multinomial variable".format(var))
        if len(var.values) < 2 or \
                multinomial_treatment == Continuize.Remove or \
                (multinomial_treatment == Continuize.RemoveMultinomial
                 and len(var.values) > 2):
            return []
        elif multinomial_treatment == Continuize.AsOrdinal:
            return [ordinal_to_continuous(var)]
        elif multinomial_treatment == Continuize.AsNormalizedOrdinal:
            return [ordinal_to_norm_continuous(var)]
        elif multinomial_treatment == Continuize.Indicators:
            return one_hot_coding(var)
        elif multinomial_treatment in (
                Continuize.FirstAsBase, Continuize.RemoveMultinomial):
            return dummy_coding(var)
        elif multinomial_treatment == Continuize.FrequentAsBase:
            dist = _ensure_dist(var, data_or_dist)
            modus = dist.modus()
            return dummy_coding(var, base_value=modus)
        elif multinomial_treatment == Continuize.Leave:
            return [var]
        raise ValueError("Invalid value of `multinomial_treatment`")

    if var.is_continuous:
        return continuize_continuous()
    elif var.is_discrete:
        return continuize_discrete()
    raise TypeError("Non-primitive variables cannot be continuized")


def _ensure_dist(var, data_or_dist):
    if isinstance(data_or_dist, distribution.Discrete):
        if not var.is_discrete:
            raise TypeError
        return data_or_dist
    elif isinstance(data_or_dist, distribution.Continuous):
        if not var.is_continuous:
            raise TypeError
        return data_or_dist
    elif isinstance(data_or_dist, Orange.data.Storage):
        return distribution.get_distribution(data_or_dist, var)
    else:
        raise ValueError("Need a distribution or data.")


def normalized_var(var, translate, scale):
    return Orange.data.ContinuousVariable(var.name,
                                          compute_value=Normalizer(var, translate, scale))


def ordinal_to_continuous(var):
    return Orange.data.ContinuousVariable(var.name,
                                          compute_value=Identity(var))


def ordinal_to_norm_continuous(var):
    n_values = len(var.values)
    return normalized_var(var, 0, 1 / (n_values - 1))


def normalize_by_sd(var, dist):
    mean, sd = dist.mean(), dist.standard_deviation()
    sd = sd if sd > 1e-10 else 1
    return normalized_var(var, mean, 1 / sd)


def center_to_mean(var, dist):
    return normalized_var(var, dist.mean(), 1)


def divide_by_sd(var, dist):
    sd = dist.standard_deviation()
    sd = sd if sd > 1e-10 else 1
    return normalized_var(var, 0, 1 / sd)


def normalize_to_11(var, dist):
    return normalize_by_span(var, dist, False)


def normalize_to_01(var, dist):
    return normalize_by_span(var, dist, True)


def normalize_by_span(var, dist, zero_based=True):
    v_max, v_min = dist.max(), dist.min()
    span = (v_max - v_min)
    if span < 1e-15:
        span = 1
    if zero_based:
        return normalized_var(var, v_min, 1 / span)
    else:
        return normalized_var(var, (v_min + v_max) / 2, 2 / span)


class DomainContinuizer(Reprable):
    def __init__(self,
                 multinomial_treatment=Continuize.Indicators,
                 continuous_treatment=OWContinuize.Normalize.Leave,
                 class_treatment=Continuize.Leave):
        self.multinomial_treatment = multinomial_treatment
        self.continuous_treatment = continuous_treatment
        self.class_treatment = class_treatment

    def __call__(self, data):
        treat = self.multinomial_treatment
        domain = data.domain
        if (treat == Continuize.ReportError and
                any(var.is_discrete and len(var.values) > 2 for var in domain)):
            raise ValueError("Domain has multinomial attributes")

        newdomain = continuize_domain(
            data,
            self.multinomial_treatment,
            self.continuous_treatment,
            self.class_treatment)
        return newdomain


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWContinuize).run(Table("iris"))
