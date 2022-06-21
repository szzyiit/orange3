import numbers

import numpy
from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.data.util import get_unique_names
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from Orange.preprocess import preprocess
from Orange.projection import PCA
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.slidergraph import SliderGraph
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output

import numpy as np


# Maximum number of PCA components that we can set in the widget
MAX_COMPONENTS = 100
LINE_NAMES = ["component variance", "cumulative variance"]


class OWPCA(widget.OWWidget):
    name = "主成分分析(PCA)"
    description = "主成分分析与 scree图。请只连接数值数据"
    icon = "icons/PCA.svg"
    priority = 3050
    keywords = ["principal component analysis", "linear transformation", 'zhuchengfen']
    category = '非监督(Unsupervised)'

    class Inputs:
        data = Input("数据(Data)", Table, replaces=['Data'])

    class Outputs:
        transformed_data = Output("转换的数据(Transformed Data)", Table, replaces=["Transformed data", "Transformed Data"])
        data = Output("数据(Data)", Table, default=True, replaces=['Data'])
        components = Output("成分(Components)", Table, replaces=['Components'])
        pca = Output("主成分分析(PCA)", PCA, dynamic=False, replaces=['PCA'])
        reconstructed = Output("重构的数据(Reconstructed Data)", Table, replaces=['Reconstructed data', 'Reconstructed Data'])
        residual = Output('残差平方(Residual square)', Table, replaces=['Residual square'])
        sum_residual = Output('残差平方和(Sum of residual square)', Table, replaces=['Sum of residual square'])

    ncomponents = settings.Setting(2)
    variance_covered = settings.Setting(100)
    auto_commit = settings.Setting(True)
    normalize = settings.Setting(True)
    maxp = settings.Setting(20)
    axis_labels = settings.Setting(10)

    graph_name = "plot.plotItem"

    class Warning(widget.OWWidget.Warning):
        trivial_components = widget.Msg(
            "All components of the PCA are trivial (explain 0 variance). "
            "Input data is constant (or near constant).")

    class Error(widget.OWWidget.Error):
        no_features = widget.Msg("At least 1 feature is required")
        no_instances = widget.Msg("At least 1 data instance is required")

    def __init__(self):
        super().__init__()
        self.data = None

        self._pca = None
        self._transformed = None
        self._variance_ratio = None
        self._cumulative = None
        self._init_projector()

        # Components Selection
        form = QFormLayout()
        box = gui.widgetBox(self.controlArea, "成分选择",
                            orientation=form)

        self.components_spin = gui.spin(
            box, self, "ncomponents", 1, MAX_COMPONENTS,
            callback=self._update_selection_component_spin,
            keyboardTracking=False, addToLayout=False
        )
        self.components_spin.setSpecialValueText("All")

        self.variance_spin = gui.spin(
            box, self, "variance_covered", 1, 100,
            callback=self._update_selection_variance_spin,
            keyboardTracking=False, addToLayout=False
        )
        self.variance_spin.setSuffix("%")

        form.addRow("成分:", self.components_spin)
        form.addRow("覆盖的贡献(Variance covered):", self.variance_spin)

        # Options
        self.options_box = gui.vBox(self.controlArea, "选项")
        # 可能有bug，先不要用，如果需要归一化用预处理器去做
        # self.normalize_box = gui.checkBox(
        #     self.options_box, self, "normalize",
        #     "归一化数据", callback=self._update_normalize,
        #     attribute=Qt.WA_LayoutUsesWidgetRect
        # )

        self.maxp_spin = gui.spin(
            self.options_box, self, "maxp", 1, MAX_COMPONENTS,
            label="只显示前...个", callback=self._setup_plot,
            keyboardTracking=False
        )

        gui.rubber(self.controlArea)

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

        self.plot = SliderGraph(
            "主成分", "贡献率(Proportion of variance)",
            self._on_cut_changed)

        self.mainArea.layout().addWidget(self.plot)
        # self._update_normalize()

    @Inputs.data
    def set_data(self, data):
        self.clear_messages()
        self.clear()
        self.information()
        self.data = None
        if not data:
            self.clear_outputs()
        if isinstance(data, SqlTable):
            if data.approx_len() < AUTO_DL_LIMIT:
                data = Table(data)
            else:
                self.information("Data has been sampled")
                data_sample = data.sample_time(1, no_cache=True)
                data_sample.download_data(2000, partial=True)
                data = Table(data_sample)
        if isinstance(data, Table):
            if not data.domain.attributes:
                self.Error.no_features()
                self.clear_outputs()
                return
            if not data:
                self.Error.no_instances()
                self.clear_outputs()
                return

        self._init_projector()

        self.data = data
        self.fit()

    def fit(self):
        self.clear()
        self.Warning.trivial_components.clear()
        if self.data is None:
            return

        data = self.data

        self.normalize = False
        if self.normalize:
            self._pca_projector.preprocessors = \
                self._pca_preprocessors + [preprocess.Normalize(center=False)]
        else:
            self._pca_projector.preprocessors = self._pca_preprocessors

        if not isinstance(data, SqlTable):
            pca = self._pca_projector(data)
            variance_ratio = pca.explained_variance_ratio_
            cumulative = numpy.cumsum(variance_ratio)

            if numpy.isfinite(cumulative[-1]):
                self.components_spin.setRange(0, len(cumulative))
                self._pca = pca
                self._variance_ratio = variance_ratio
                self._cumulative = cumulative
                self._setup_plot()
            else:
                self.Warning.trivial_components()

            self.commit.now()

    def clear(self):
        self._pca = None
        self._transformed = None
        self._variance_ratio = None
        self._cumulative = None
        self.plot.clear_plot()

    def clear_outputs(self):
        self.Outputs.transformed_data.send(None)
        self.Outputs.data.send(None)
        self.Outputs.components.send(None)
        self.Outputs.pca.send(self._pca_projector)

    def _setup_plot(self):
        if self._pca is None:
            self.plot.clear_plot()
            return

        explained_ratio = self._variance_ratio
        explained = self._cumulative
        cutpos = self._nselected_components()
        p = min(len(self._variance_ratio), self.maxp)

        self.plot.update(
            numpy.arange(1, p+1), [explained_ratio[:p], explained[:p]],
            [Qt.red, Qt.darkYellow], cutpoint_x=cutpos, names=LINE_NAMES)

        self._update_axis()

    def _on_cut_changed(self, components):
        if components == self.ncomponents \
                or self.ncomponents == 0:
            return

        self.ncomponents = components
        if self._pca is not None:
            var = self._cumulative[components - 1]
            if numpy.isfinite(var):
                self.variance_covered = int(var * 100)

        self._invalidate_selection()

    def _update_selection_component_spin(self):
        # cut changed by "ncomponents" spin.
        if self._pca is None:
            self._invalidate_selection()
            return

        if self.ncomponents == 0:
            # Special "All" value
            cut = len(self._variance_ratio)
        else:
            cut = self.ncomponents

        var = self._cumulative[cut - 1]
        if numpy.isfinite(var):
            self.variance_covered = int(var * 100)

        self.plot.set_cut_point(cut)
        self._invalidate_selection()

    def _update_selection_variance_spin(self):
        # cut changed by "max variance" spin.
        if self._pca is None:
            return

        cut = numpy.searchsorted(self._cumulative,
                                 self.variance_covered / 100.0) + 1
        cut = min(cut, len(self._cumulative))
        self.ncomponents = cut
        self.plot.set_cut_point(cut)
        self._invalidate_selection()

    # def _update_normalize(self):
    #     self.fit()
    #     if self.data is None:
    #         self._invalidate_selection()

    def _init_projector(self):
        self._pca_projector = PCA(n_components=MAX_COMPONENTS, random_state=0)
        self._pca_projector.component = self.ncomponents
        self._pca_preprocessors = PCA.preprocessors

    def _nselected_components(self):
        """Return the number of selected components."""
        if self._pca is None:
            return 0

        if self.ncomponents == 0:
            # Special "All" value
            max_comp = len(self._variance_ratio)
        else:
            max_comp = self.ncomponents

        var_max = self._cumulative[max_comp - 1]
        if var_max != numpy.floor(self.variance_covered / 100.0):
            cut = max_comp
            assert numpy.isfinite(var_max)
            self.variance_covered = int(var_max * 100)
        else:
            self.ncomponents = cut = numpy.searchsorted(
                self._cumulative, self.variance_covered / 100.0) + 1
        return cut

    def _invalidate_selection(self):
        self.commit.deferred()

    def _update_axis(self):
        p = min(len(self._variance_ratio), self.maxp)
        axis = self.plot.getAxis("bottom")
        d = max((p-1)//(self.axis_labels-1), 1)
        axis.setTicks([[(i, str(i)) for i in range(1, p + 1, d)]])

    @gui.deferred
    def commit(self):
        transformed = data = components = reconstructed = residual = sum_residual = None
        if self._pca is not None:
            if self._transformed is None:
                # Compute the full transform (MAX_COMPONENTS components) once.
                self._transformed = self._pca(self.data)
            transformed = self._transformed

            if self._variance_ratio is not None:
                for var, explvar in zip(
                        transformed.domain.attributes,
                        self._variance_ratio[:self.ncomponents]):
                    var.attributes["variance"] = round(explvar, 6)
            domain = Domain(
                transformed.domain.attributes[:self.ncomponents],
                self.data.domain.class_vars,
                self.data.domain.metas
            )
            transformed = transformed.from_table(domain, transformed)

            # prevent caching new features by defining compute_value
            proposed = [a.name for a in self._pca.orig_domain.attributes]
            meta_name = get_unique_names(proposed, 'components')
            meta_vars = [StringVariable(name=meta_name)]
            metas = numpy.array([['PC{}'.format(i + 1)
                                  for i in range(self.ncomponents)]],
                                dtype=object).T
            if self._variance_ratio is not None:
                variance_name = get_unique_names(proposed, "variance")
                meta_vars.append(ContinuousVariable(variance_name))
                metas = numpy.hstack(
                    (metas,
                     self._variance_ratio[:self.ncomponents, None]))

            dom = Domain(
                [ContinuousVariable(name, compute_value=lambda _: None)
                 for name in proposed],
                metas=meta_vars)
            components = Table(dom, self._pca.components_[:self.ncomponents],
                               metas=metas)
            components.name = '成分'

            data_dom = Domain(
                self.data.domain.attributes,
                self.data.domain.class_vars,
                self.data.domain.metas + domain.attributes)
            data = Table.from_numpy(
                data_dom, self.data.X, self.data.Y,
                numpy.hstack((self.data.metas, transformed.X)),
                ids=self.data.ids)
            data.name = '数据'
            reconstructed_data = transformed.X @ components.X + self._pca.mean_

            reconstructed = Table.from_numpy(data_dom, reconstructed_data, self.data.Y,
                            numpy.hstack((self.data.metas, transformed.X)),
                            ids=self.data.ids)
            reconstructed.name = '重构的数据'
            residual_data = (self.data.X - reconstructed_data)**2
            residual = Table.from_numpy(data_dom, residual_data, self.data.Y,
                            numpy.hstack((self.data.metas, transformed.X)),
                            ids=self.data.ids)
            residual.name = '残差平方'

            sum_residual_data = np.sum(residual_data, axis=1)
            r_dom = Domain(
                [ContinuousVariable('sum_residual')])
            sum_residual = Table.from_numpy(r_dom, sum_residual_data.reshape(-1, 1))

        self._pca_projector.component = self.ncomponents
        self.Outputs.transformed_data.send(transformed)
        self.Outputs.components.send(components)
        self.Outputs.data.send(data)
        self.Outputs.pca.send(self._pca_projector)
        self.Outputs.reconstructed.send(reconstructed)
        self.Outputs.residual.send(residual)
        self.Outputs.sum_residual.send(sum_residual)

    def send_report(self):
        if self.data is None:
            return
        self.report_items((
            ("Normalize data", str(self.normalize)),
            ("Selected components", self.ncomponents),
            ("Explained variance", "{:.3f} %".format(self.variance_covered))
        ))
        self.report_plot()

    @classmethod
    def migrate_settings(cls, settings, version):
        if "variance_covered" in settings:
            # Due to the error in gh-1896 the variance_covered was persisted
            # as a NaN value, causing a TypeError in the widgets `__init__`.
            vc = settings["variance_covered"]
            if isinstance(vc, numbers.Real):
                if numpy.isfinite(vc):
                    vc = int(vc)
                else:
                    vc = 100
                settings["variance_covered"] = vc
        if settings.get("ncomponents", 0) > MAX_COMPONENTS:
            settings["ncomponents"] = MAX_COMPONENTS

        # Remove old `decomposition_idx` when SVD was still included
        settings.pop("decomposition_idx", None)

        # Remove RemotePCA settings
        settings.pop("batch_size", None)
        settings.pop("address", None)
        settings.pop("auto_update", None)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPCA).run(Table("housing"))
