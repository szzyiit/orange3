import enum
import math
import itertools

from typing import (
    Iterable, Mapping, Any, Optional, Union, TypeVar, Type, NamedTuple,
    Sequence, Tuple
)

import numpy as np
import scipy.sparse as sp

from AnyQt.QtWidgets import (
    QSizePolicy, QGraphicsScene, QGraphicsView, QGraphicsRectItem,
    QGraphicsWidget, QGraphicsSimpleTextItem, QGraphicsPixmapItem,
    QGraphicsGridLayout, QGraphicsLinearLayout, QGraphicsLayoutItem,
    QFormLayout, QApplication, QComboBox, QWIDGETSIZE_MAX
)
from AnyQt.QtGui import (
    QFontMetrics, QPen, QPixmap, QTransform,
    QStandardItemModel, QStandardItem,
)
from AnyQt.QtCore import (
    Qt, QSize, QPointF, QSizeF, QRectF, QObject, QEvent,
    pyqtSignal as Signal,
)
import pyqtgraph as pg

from orangewidget.utils.combobox import ComboBox

from Orange.data import Domain, Table
from Orange.data.sql.table import SqlTable
import Orange.distance

from Orange.clustering import hierarchical, kmeans
from Orange.widgets.utils import colorpalettes
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.stickygraphicsview import StickyGraphicsView
from Orange.widgets.utils.graphicstextlist import scaled, TextListWidget
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                ANNOTATED_DATA_SIGNAL_Chinese_NAME,
                                                ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets import widget, gui, settings
from Orange.widgets.unsupervised.owhierarchicalclustering import \
    DendrogramWidget
from Orange.widgets.unsupervised.owdistancemap import TextList as TextListWidget
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Input, Output


def kmeans_compress(X, k=50):
    km = kmeans.KMeans(n_clusters=k, n_init=5, random_state=42)
    return km.get_model(X)


def leaf_indices(tree):
    return [leaf.value.index for leaf in hierarchical.leaves(tree)]


def levels_with_thresholds(low, high, threshold_low, threshold_high, center_palette):
    lt = low + (high - low) * threshold_low
    ht = low + (high - low) * threshold_high
    if center_palette:
        ht = max(abs(lt), abs(ht))
        lt = -max(abs(lt), abs(ht))
    return lt, ht

# TODO:
#     * Richer Tool Tips
#     * Color map edit/manage
#     * Restore saved row selection (?)
#     * 'namespace' use cleanup

# Heatmap grid description
# ########################
#
# Heatmaps can be split vertically (by discrete var) and/or horizontaly
# (by suitable variable labels).
# Each vertical split has a title (split variable value) and can
# be sorted/clustred individually. Horizontal splits can also be
# clustered but will share the same cluster)


class RowPart(NamedTuple):
    """
    A row group

    Attributes
    ----------
    title: str
        Group title
    indices : (N, ) int ndarray | slice
        Indexes the input data to retrieve the row subset for the group.
    cluster : hierarchical.Tree optional
    cluster_ordered : hierarchical.Tree optional
    """
    title: str
    indices: Sequence[int]
    cluster: Optional[hierarchical.Tree] = None
    cluster_ordered: Optional[hierarchical.Tree] = None

    @property
    def can_cluster(self):
        if isinstance(self.indices, slice):
            return (self.indices.stop - self.indices.start) > 1
        else:
            return len(self.indices) > 1

    @property
    def cluster_ord(self):
        return self.cluster_ordered


class ColumnPart(NamedTuple):
    """
    A column group

    Attributes
    ----------
    title : str
        Column group title
    indices : (N, ) int ndarray | slice
        Indexes the input data to retrieve the column subset for the group.
    domain : List[Variable]
        List of variables in the group.
    cluster : hierarchical.Tree optional
    cluster_ordered : hierarchical.Tree optional
    """
    title: Optional[str]
    indices: Sequence[int]
    domain: Sequence[Orange.data.Variable]
    cluster: Optional[hierarchical.Tree] = None
    cluster_ordered: Optional[hierarchical.Tree] = None

    @property
    def cluster_ord(self):
        return self.cluster_ordered


class Parts(NamedTuple):
    rows: Sequence[RowPart]        #: A list of RowPart descriptors
    columns: Sequence[ColumnPart]  #: A list of ColumnPart descriptors
    span: Tuple[float, float]      #: (min, max) global data range

    levels = property(lambda self: self.span)


def cbselect(cb: QComboBox, value, role: Qt.ItemDataRole = Qt.EditRole) -> None:
    """
    Find and select the `value` in the `cb` QComboBox.

    Parameters
    ----------
    cb: QComboBox
    value: Any
    role: Qt.ItemDataRole
        The data role in the combo box model to match value against
    """
    cb.setCurrentIndex(cb.findData(value, role))


class Clustering(enum.IntEnum):
    #: No clustering
    None_ = 0
    #: Hierarchical clustering
    Clustering = 1
    #: Hierarchical clustering with optimal leaf ordering
    OrderedClustering = 2


ClusteringRole = Qt.UserRole + 13
#: Item data for clustering method selection models
ClusteringModelData = [
    {
        Qt.DisplayRole: "None",
        Qt.ToolTipRole: "No clustering",
        ClusteringRole: Clustering.None_,
    }, {
        Qt.DisplayRole: "Clustering",
        Qt.ToolTipRole: "Apply hierarchical clustering",
        ClusteringRole: Clustering.Clustering,
    }, {
        Qt.DisplayRole: "Clustering (opt. ordering)",
        Qt.ToolTipRole: "Apply hierarchical clustering with optimal leaf "
                        "ordering.",
        ClusteringRole: Clustering.OrderedClustering,
    }
]


def create_list_model(
        items: Iterable[Mapping[Qt.ItemDataRole, Any]],
        parent: Optional[QObject] = None,
) -> QStandardItemModel:
    """Create list model from an item date iterable."""
    model = QStandardItemModel(parent)
    for item in items:
        sitem = QStandardItem()
        for role, value in item.items():
            sitem.setData(value, role)
        model.appendRow([sitem])
    return model


E = TypeVar("E", bound=enum.Enum)  # pylint: disable=invalid-name


def enum_get(etype: Type[E], name: str, default: E) -> E:
    """
    Return an Enum member by `name`. If no such member exists in `etype`
    return `default`.
    """
    try:
        return etype[name]
    except LookupError:
        return default


class OWHeatMap(widget.OWWidget):
    name = "热图(Heat Map)"
    description = "为一对属性绘制热图。"
    icon = "icons/Heatmap.svg"
    priority = 260
    keywords = []

    class Inputs:
        data = Input("数据(Data)", Table, replaces=['Data'])

    class Outputs:
        selected_data = Output("选定的数据(Selected Data)", Table, default=True, replaces=['Selected Data'])
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_Chinese_NAME, Table, replaces=['Data'])

    settings_version = 3

    settingsHandler = settings.DomainContextHandler()

    NoPosition, PositionTop, PositionBottom = 0, 1, 2

    # Disable clustering for inputs bigger than this
    MaxClustering = 25000
    # Disable cluster leaf ordering for inputs bigger than this
    MaxOrderedClustering = 1000

    threshold_low = settings.Setting(0.0)
    threshold_high = settings.Setting(1.0)

    merge_kmeans = settings.Setting(False)
    merge_kmeans_k = settings.Setting(50)

    # Display stripe with averages
    averages = settings.Setting(True)
    # Display legend
    legend = settings.Setting(True)
    # Annotations
    annotation_var = settings.ContextSetting(None)
    # Discrete variable used to split that data/heatmaps (vertically)
    split_by_var = settings.ContextSetting(None)

    # Selected row/column clustering method (name)
    col_clustering_method: str = settings.Setting(Clustering.None_.name)
    row_clustering_method: str = settings.Setting(Clustering.None_.name)

    palette_name = settings.Setting(colorpalettes.DefaultContinuousPaletteName)
    column_label_pos = settings.Setting(PositionTop)
    selected_rows = settings.Setting(None, schema_only=True)

    auto_commit = settings.Setting(True)

    graph_name = "scene"

    left_side_scrolling = True

    class Information(widget.OWWidget.Information):
        sampled = Msg("Data has been sampled")
        discrete_ignored = Msg("{} categorical feature{} ignored")
        row_clust = Msg("{}")
        col_clust = Msg("{}")
        sparse_densified = Msg("Showing this data may require a lot of memory")

    class Error(widget.OWWidget.Error):
        no_continuous = Msg("No numeric features")
        not_enough_features = Msg("Not enough features for column clustering")
        not_enough_instances = Msg("Not enough instances for clustering")
        not_enough_instances_k_means = Msg(
            "Not enough instances for k-means merging")
        not_enough_memory = Msg("Not enough memory to show this data")

    class Warning(widget.OWWidget.Warning):
        empty_clusters = Msg("Empty clusters were removed")

    def __init__(self):
        super().__init__()
        self.__pending_selection = self.selected_rows

        # A kingdom for a save_state/restore_state
        self.col_clustering = enum_get(
            Clustering, self.col_clustering_method, Clustering.None_)
        self.row_clustering = enum_get(
            Clustering, self.row_clustering_method, Clustering.None_)

        @self.settingsAboutToBePacked.connect
        def _():
            self.col_clustering_method = self.col_clustering.name
            self.row_clustering_method = self.row_clustering.name

        # set default settings
        self.space_x = 10

        self.colorSettings = None
        self.selectedSchemaIndex = 0

        self.palette = None
        self.keep_aspect = False

        #: The original data with all features (retained to
        #: preserve the domain on the output)
        self.input_data = None
        #: The effective data striped of discrete features, and often
        #: merged using k-means
        self.data = None
        self.effective_data = None
        #: kmeans model used to merge rows of input_data
        self.kmeans_model = None
        #: merge indices derived from kmeans
        #: a list (len==k) of int ndarray where the i-th item contains
        #: the indices which merge the input_data into the heatmap row i
        self.merge_indices = None

        self.__rows_cache = {}
        self.__columns_cache = {}

        # GUI definition
        colorbox = gui.vBox(self.controlArea, "颜色")
        self.color_cb = gui.palette_combo_box(self.palette_name)
        self.color_cb.currentIndexChanged.connect(self.update_color_schema)
        colorbox.layout().addWidget(self.color_cb)

        # TODO: Add 'Manage/Add/Remove' action.

        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )

        lowslider = gui.hSlider(
            colorbox, self, "threshold_low", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_lowslider)
        highslider = gui.hSlider(
            colorbox, self, "threshold_high", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_highslider)

        form.addRow("低:", lowslider)
        form.addRow("高:", highslider)

        colorbox.layout().addLayout(form)

        mergebox = gui.vBox(self.controlArea, "合并",)
        gui.checkBox(mergebox, self, "merge_kmeans", "以k-均值方式合并",
                     callback=self.__update_row_clustering)
        ibox = gui.indentedBox(mergebox)
        gui.spin(ibox, self, "merge_kmeans_k", minv=5, maxv=500,
                 label="聚类:", keyboardTracking=False,
                 callbackOnReturn=True, callback=self.update_merge)

        cluster_box = gui.vBox(self.controlArea, "聚类")
        # Row clustering
        self.row_cluster_cb = cb = ComboBox(maximumContentsLength=14)
        cb.setModel(create_list_model(ClusteringModelData, self))
        cbselect(cb, self.row_clustering, ClusteringRole)
        self.connect_control(
            "row_clustering",
            lambda value, cb=cb: cbselect(cb, value, ClusteringRole)
        )
        @cb.activated.connect
        def _(idx, cb=cb):
            self.set_row_clustering(cb.itemData(idx, ClusteringRole))

        # Column clustering
        self.col_cluster_cb = cb = ComboBox(maximumContentsLength=14)
        cb.setModel(create_list_model(ClusteringModelData, self))
        cbselect(cb, self.col_clustering, ClusteringRole)
        self.connect_control(
            "col_clustering",
            lambda value, cb=cb: cbselect(cb, value, ClusteringRole)
        )
        @cb.activated.connect
        def _(idx, cb=cb):
            self.set_col_clustering(cb.itemData(idx, ClusteringRole))

        form = QFormLayout(
            labelAlignment=Qt.AlignLeft, formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
        )
        form.addRow("行:", self.row_cluster_cb)
        form.addRow("列:", self.col_cluster_cb)
        cluster_box.layout().addLayout(form)
        box = gui.vBox(self.controlArea, "以...划分")

        self.row_split_model = DomainModel(
            placeholder="(无)",
            valid_types=(Orange.data.DiscreteVariable,),
            parent=self,
        )
        self.row_split_cb = cb = ComboBox(
            enabled=not self.merge_kmeans,
            sizeAdjustPolicy=ComboBox.AdjustToMinimumContentsLengthWithIcon,
            minimumContentsLength=14,
            toolTip="Split the heatmap vertically by a categorical column"
        )
        self.row_split_cb.setModel(self.row_split_model)
        self.connect_control(
            "split_by_var", lambda value, cb=cb: cbselect(cb, value)
        )
        self.connect_control(
            "merge_kmeans", self.row_split_cb.setDisabled
        )
        self.split_by_var = None

        self.row_split_cb.activated.connect(
            self.__on_split_rows_activated
        )
        box.layout().addWidget(self.row_split_cb)

        box = gui.vBox(self.controlArea, '注释和图例')

        gui.checkBox(box, self, 'legend', '显示图例',
                     callback=self.update_legend)

        gui.checkBox(box, self, 'averages', '带平均线的条纹',
                     callback=self.update_averages_stripe)

        annotbox = gui.vBox(box, "行注释", addSpace=False)
        annotbox.setFlat(True)
        self.annotation_model = DomainModel(placeholder="(None)")
        gui.comboBox(
            annotbox, self, "annotation_var", contentsLength=12,
            model=self.annotation_model, callback=self.update_annotations)

        posbox = gui.vBox(box, "列标签位置", addSpace=False)
        posbox.setFlat(True)

        gui.comboBox(
            posbox, self, "column_label_pos",
            items=["无", "顶部", "底部", "顶部和底部"],
            callback=self.update_column_annotations)

        gui.checkBox(self.controlArea, self, "keep_aspect",
                     "保持纵横比", box="调整大小",
                     callback=self.__aspect_mode_changed)

        gui.rubber(self.controlArea)
        gui.auto_send(self.controlArea, self, "auto_commit")

        # Scene with heatmap
        self.heatmap_scene = self.scene = HeatmapScene(parent=self)
        self.selection_manager = HeatmapSelectionManager(self)
        self.selection_manager.selection_changed.connect(
            self.__update_selection_geometry)
        self.selection_manager.selection_finished.connect(
            self.on_selection_finished)
        self.heatmap_scene.set_selection_manager(self.selection_manager)

        item = QGraphicsRectItem(0, 0, 10, 10, None)
        self.heatmap_scene.addItem(item)
        self.heatmap_scene.itemsBoundingRect()
        self.heatmap_scene.removeItem(item)

        self.sceneView = StickyGraphicsView(
            self.scene,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            viewportUpdateMode=QGraphicsView.FullViewportUpdate,
        )

        self.sceneView.viewport().installEventFilter(self)

        self.mainArea.layout().addWidget(self.sceneView)
        self.heatmap_scene.widget = None

        self.heatmap_widget_grid = [[]]
        self.attr_annotation_widgets = []
        self.attr_dendrogram_widgets = []
        self.gene_annotation_widgets = []
        self.gene_dendrogram_widgets = []

        self.selection_rects = []
        self.selected_rows = []

    @property
    def center_palette(self):
        palette = self.color_cb.currentData()
        return bool(palette.flags & palette.Diverging)

    def set_row_clustering(self, method: Clustering) -> None:
        assert isinstance(method, Clustering)
        if self.row_clustering != method:
            self.row_clustering = method
            cbselect(self.row_cluster_cb, method, ClusteringRole)
            self.__update_row_clustering()

    def set_col_clustering(self, method: Clustering) -> None:
        assert isinstance(method, Clustering)
        if self.col_clustering != method:
            self.col_clustering = method
            cbselect(self.col_cluster_cb, method, ClusteringRole)
            self.__update_column_clustering()

    def sizeHint(self):
        return QSize(800, 400)

    def color_palette(self):
        return self.color_cb.currentData().lookup_table()

    def clear(self):
        self.data = None
        self.input_data = None
        self.effective_data = None
        self.kmeans_model = None
        self.merge_indices = None
        self.annotation_model.set_domain(None)
        self.annotation_var = None
        self.row_split_model.set_domain(None)
        self.split_by_var = None
        self.clear_scene()
        self.selected_rows = []
        self.__columns_cache.clear()
        self.__rows_cache.clear()
        self.__update_clustering_enable_state(None)

    def clear_scene(self):
        self.selection_manager.set_heatmap_widgets([[]])
        self.heatmap_scene.clear()
        self.heatmap_scene.widget = None
        self.heatmap_widget_grid = [[]]
        self.col_annotation_widgets = []
        self.col_annotation_widgets_bottom = []
        self.col_annotation_widgets_top = []
        self.row_annotation_widgets = []
        self.col_dendrograms = []
        self.row_dendrograms = []
        self.selection_rects = []
        self.sceneView.setSceneRect(QRectF())
        self.sceneView.setHeaderSceneRect(QRectF())
        self.sceneView.setFooterSceneRect(QRectF())

    @Inputs.data
    def set_dataset(self, data=None):
        """Set the input dataset to display."""
        self.closeContext()
        self.clear()
        self.clear_messages()

        if isinstance(data, SqlTable):
            if data.approx_len() < 4000:
                data = Table(data)
            else:
                self.Information.sampled()
                data_sample = data.sample_time(1, no_cache=True)
                data_sample.download_data(2000, partial=True)
                data = Table(data_sample)

        if data is not None and not len(data):
            data = None

        if data is not None and sp.issparse(data.X):
            try:
                data = data.to_dense()
            except MemoryError:
                data = None
                self.Error.not_enough_memory()
            else:
                self.Information.sparse_densified()

        input_data = data

        # Data contains no attributes or meta attributes only
        if data is not None and len(data.domain.attributes) == 0:
            self.Error.no_continuous()
            input_data = data = None

        # Data contains some discrete attributes which must be filtered
        if data is not None and \
                any(var.is_discrete for var in data.domain.attributes):
            ndisc = sum(var.is_discrete for var in data.domain.attributes)
            data = data.transform(
                Domain([var for var in data.domain.attributes
                        if var.is_continuous],
                       data.domain.class_vars,
                       data.domain.metas))
            if not data.domain.attributes:
                self.Error.no_continuous()
                input_data = data = None
            else:
                self.Information.discrete_ignored(
                    ndisc, "s" if ndisc > 1 else "")

        self.data = data
        self.input_data = input_data

        if data is not None:
            self.annotation_model.set_domain(self.input_data.domain)
            self.annotation_var = None
            self.row_split_model.set_domain(data.domain)
            if data.domain.has_discrete_class:
                self.split_by_var = data.domain.class_var
            else:
                self.split_by_var = None
            self.openContext(self.input_data)
            if self.split_by_var not in self.row_split_model:
                self.split_by_var = None

        self.update_heatmaps()
        if data is not None and self.__pending_selection is not None:
            self.selection_manager.select_rows(self.__pending_selection)
            self.selected_rows = self.__pending_selection
            self.__pending_selection = None

        self.unconditional_commit()

    def __on_split_rows_activated(self):
        self.set_split_variable(self.row_split_cb.currentData(Qt.EditRole))

    def set_split_variable(self, var):
        if var != self.split_by_var:
            self.split_by_var = var
            self.update_heatmaps()

    def update_heatmaps(self):
        if self.data is not None:
            self.clear_scene()
            self.clear_messages()
            if self.col_clustering != Clustering.None_ and \
                    len(self.data.domain.attributes) < 2:
                self.Error.not_enough_features()
            elif (self.col_clustering != Clustering.None_ or
                  self.row_clustering != Clustering.None_) and \
                    len(self.data) < 2:
                self.Error.not_enough_instances()
            elif self.merge_kmeans and len(self.data) < 3:
                self.Error.not_enough_instances_k_means()
            else:
                self.heatmapparts = self.construct_heatmaps(
                    self.data, self.split_by_var
                )
                self.construct_heatmaps_scene(
                    self.heatmapparts, self.effective_data)
                self.selected_rows = []
        else:
            self.clear()

    def update_merge(self):
        self.kmeans_model = None
        self.merge_indices = None
        if self.data is not None and self.merge_kmeans:
            self.update_heatmaps()
            self.commit()

    def _make_parts(self, data, group_var=None):
        """
        Make initial `Parts` for data, split by group_var, group_key
        """
        if group_var is not None:
            assert group_var.is_discrete
            _col_data, _ = data.get_column_view(group_var)
            row_indices = [np.flatnonzero(_col_data == i)
                           for i in range(len(group_var.values))]
            row_groups = [RowPart(title=name, indices=ind,
                                  cluster=None, cluster_ordered=None)
                          for name, ind in zip(group_var.values, row_indices)]
        else:
            row_groups = [RowPart(title=None, indices=slice(0, len(data)),
                                  cluster=None, cluster_ordered=None)]

        col_groups = [
            ColumnPart(
                title=None, indices=slice(0, len(data.domain.attributes)),
                domain=data.domain, cluster=None, cluster_ordered=None)
        ]

        minv, maxv = np.nanmin(data.X), np.nanmax(data.X)
        return Parts(row_groups, col_groups, span=(minv, maxv))

    def cluster_rows(self, data: Table, parts: Parts, ordered=False) -> Parts:
        row_groups = []
        for row in parts.rows:
            if row.cluster is not None:
                cluster = row.cluster
            else:
                cluster = None
            if row.cluster_ordered is not None:
                cluster_ord = row.cluster_ordered
            else:
                cluster_ord = None

            if row.can_cluster:
                matrix = None
                need_dist = cluster is None or (ordered and cluster_ord is None)
                if need_dist:
                    subset = data[row.indices]
                    matrix = Orange.distance.Euclidean(subset)

                if cluster is None:
                    assert len(matrix) < self.MaxClustering
                    cluster = hierarchical.dist_matrix_clustering(
                        matrix, linkage=hierarchical.WARD
                    )
                if ordered and cluster_ord is None:
                    assert len(matrix) < self.MaxOrderedClustering
                    cluster_ord = hierarchical.optimal_leaf_ordering(
                        cluster, matrix,
                    )
            row_groups.append(row._replace(cluster=cluster, cluster_ordered=cluster_ord))

        return parts._replace(rows=row_groups)

    def cluster_columns(self, data, parts, ordered=False):
        assert len(parts.columns) == 1, "columns split is no longer supported"
        assert all(var.is_continuous for var in data.domain.attributes)

        col0 = parts.columns[0]
        if col0.cluster is not None:
            cluster = col0.cluster
        else:
            cluster = None
        if col0.cluster_ord is not None:
            cluster_ord = col0.cluster_ord
        else:
            cluster_ord = None
        need_dist = cluster is None or (ordered and cluster_ord is None)

        matrix = None
        if need_dist:
            data = Orange.distance._preprocess(data)
            matrix = Orange.distance.PearsonR(data, axis=0)
            # nan values break clustering below
            matrix = np.nan_to_num(matrix)

        if cluster is None:
            assert matrix is not None
            assert len(matrix) < self.MaxClustering
            cluster = hierarchical.dist_matrix_clustering(
                matrix, linkage=hierarchical.WARD
            )
        if ordered and cluster_ord is None:
            assert len(matrix) < self.MaxOrderedClustering
            cluster_ord = hierarchical.optimal_leaf_ordering(cluster, matrix)

        col_groups = [col._replace(cluster=cluster, cluster_ordered=cluster_ord)
                      for col in parts.columns]
        return parts._replace(columns=col_groups)

    def construct_heatmaps(self, data, group_var=None) -> 'Parts':
        if self.merge_kmeans:
            if self.kmeans_model is None:
                effective_data = self.input_data.transform(
                    Orange.data.Domain(
                        [var for var in self.input_data.domain.attributes
                         if var.is_continuous],
                        self.input_data.domain.class_vars,
                        self.input_data.domain.metas))
                nclust = min(self.merge_kmeans_k, len(effective_data) - 1)
                self.kmeans_model = kmeans_compress(effective_data, k=nclust)
                effective_data.domain = self.kmeans_model.domain
                merge_indices = [np.flatnonzero(self.kmeans_model.labels == ind)
                                 for ind in range(nclust)]
                not_empty_indices = [i for i, x in enumerate(merge_indices)
                                     if len(x) > 0]
                self.merge_indices = \
                    [merge_indices[i] for i in not_empty_indices]
                if len(merge_indices) != len(self.merge_indices):
                    self.Warning.empty_clusters()
                effective_data = Orange.data.Table(
                    Orange.data.Domain(effective_data.domain.attributes),
                    self.kmeans_model.centroids[not_empty_indices]
                )
            else:
                effective_data = self.effective_data

            group_var = None
        else:
            self.kmeans_model = None
            self.merge_indices = None
            effective_data = data

        self.effective_data = effective_data

        self.__update_clustering_enable_state(effective_data)

        parts = self._make_parts(effective_data, group_var)
        # Restore/update the row/columns items descriptions from cache if
        # available
        rows_cache_key = (group_var,
                          self.merge_kmeans_k if self.merge_kmeans else None)
        if rows_cache_key in self.__rows_cache:
            parts = parts._replace(rows=self.__rows_cache[rows_cache_key].rows)

        if self.row_clustering != Clustering.None_:
            parts = self.cluster_rows(
                effective_data, parts,
                ordered=self.row_clustering == Clustering.OrderedClustering
            )
        if self.col_clustering != Clustering.None_:
            parts = self.cluster_columns(
                effective_data, parts,
                ordered=self.col_clustering == Clustering.OrderedClustering
            )

        # Cache the updated parts
        self.__rows_cache[rows_cache_key] = parts
        return parts

    def construct_heatmaps_scene(self, parts: Parts, data: Table) -> None:
        _T = TypeVar("_T", bound=Union[RowPart, ColumnPart])

        def select_cluster(clustering: Clustering, item: _T) -> _T:
            if clustering == Clustering.None_:
                return item._replace(cluster=None, cluster_ordered=None)
            elif clustering == Clustering.Clustering:
                return item._replace(cluster=item.cluster, cluster_ordered=None)
            elif clustering == Clustering.OrderedClustering:
                return item._replace(cluster=item.cluster_ordered, cluster_ordered=None)
            else:  # pragma: no cover
                raise TypeError()

        rows = [select_cluster(self.row_clustering, rowitem)
                for rowitem in parts.rows]
        cols = [select_cluster(self.col_clustering, colitem)
                for colitem in parts.columns]
        parts = Parts(columns=cols, rows=rows, span=parts.levels)

        self.setup_scene(parts, data)

    def setup_scene(self, parts, data):
        # parts = * a list of row descriptors (title, indices, cluster,)
        #         * a list of col descriptors (title, indices, cluster, domain)
        self.heatmap_scene.clear()
        # The top level container widget
        widget = GraphicsWidget()
        widget.layoutDidActivate.connect(self.__on_layout_activate)

        grid = QGraphicsGridLayout()
        grid.setSpacing(self.space_x)
        self.heatmap_scene.addItem(widget)

        N, M = len(parts.rows), len(parts.columns)

        # Start row/column where the heatmap items are inserted
        # (after the titles/legends/dendrograms)
        Row0 = 3
        Col0 = 3
        LegendRow = 0
        # The column for the vertical dendrogram
        DendrogramColumn = 0
        # The row for the horizontal dendrograms
        DendrogramRow = 1
        RightLabelColumn = Col0 + M
        TopLabelsRow = 2
        BottomLabelsRow = Row0 + 2 * N

        widget.setLayout(grid)

        palette = self.color_palette()

        sort_i = []
        sort_j = []

        column_dendrograms = [None] * M
        row_dendrograms = [None] * N

        for i, rowitem in enumerate(parts.rows):
            if rowitem.title:
                title = QGraphicsSimpleTextItem(rowitem.title, widget)
                item = GraphicsSimpleTextLayoutItem(title, parent=grid)
                item.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                grid.addItem(item, Row0 + i * 2, Col0)

            if rowitem.cluster:
                dendrogram = DendrogramWidget(
                    parent=widget,
                    selectionMode=DendrogramWidget.NoSelection,
                    hoverHighlightEnabled=True)
                dendrogram.set_root(rowitem.cluster)
                dendrogram.setMaximumWidth(100)
                dendrogram.setMinimumWidth(100)
                # Ignore dendrogram vertical size hint (heatmap's size
                # should define the  row's vertical size).
                dendrogram.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Ignored)
                dendrogram.itemClicked.connect(
                    lambda item, partindex=i:
                    self.__select_by_cluster(item, partindex)
                )

                grid.addItem(dendrogram, Row0 + i * 2 + 1, DendrogramColumn)
                sort_i.append(np.array(leaf_indices(rowitem.cluster)))
                row_dendrograms[i] = dendrogram
            else:
                sort_i.append(None)

        for j, colitem in enumerate(parts.columns):
            if colitem.title:
                title = QGraphicsSimpleTextItem(colitem.title, widget)
                item = GraphicsSimpleTextLayoutItem(title, parent=grid)
                grid.addItem(item, 1, Col0 + j)

            if colitem.cluster:
                dendrogram = DendrogramWidget(
                    parent=widget,
                    orientation=DendrogramWidget.Top,
                    selectionMode=DendrogramWidget.NoSelection,
                    hoverHighlightEnabled=False)

                dendrogram.set_root(colitem.cluster)
                dendrogram.setMaximumHeight(100)
                dendrogram.setMinimumHeight(100)
                # Ignore dendrogram horizontal size hint (heatmap's width
                # should define the column width).
                dendrogram.setSizePolicy(
                    QSizePolicy.Ignored, QSizePolicy.Expanding)
                grid.addItem(dendrogram, DendrogramRow, Col0 + j)
                sort_j.append(np.array(leaf_indices(colitem.cluster)))
                column_dendrograms[j] = dendrogram
            else:
                sort_j.append(None)

        heatmap_widgets = []
        for i in range(N):
            heatmap_row = []
            for j in range(M):
                row_ix = parts.rows[i].indices
                col_ix = parts.columns[j].indices
                hw = GraphicsHeatmapWidget(parent=widget)
                X_part = data[row_ix, col_ix].X

                if sort_i[i] is not None:
                    X_part = X_part[sort_i[i]]
                if sort_j[j] is not None:
                    X_part = X_part[:, sort_j[j]]

                hw.set_levels(parts.levels)
                hw.set_thresholds(self.threshold_low, self.threshold_high)
                hw.set_color_table(palette, self.center_palette)
                hw.set_show_averages(self.averages)
                hw.set_heatmap_data(X_part)

                grid.addItem(hw, Row0 + i * 2 + 1, Col0 + j)
                grid.setRowStretchFactor(Row0 + i * 2 + 1, X_part.shape[0] * 100)
                heatmap_row.append(hw)
            heatmap_widgets.append(heatmap_row)

        row_annotation_widgets = []
        col_annotation_widgets = []
        col_annotation_widgets_top = []
        col_annotation_widgets_bottom = []

        for i, rowitem in enumerate(parts.rows):
            if isinstance(rowitem.indices, slice):
                indices = np.array(
                    range(*rowitem.indices.indices(data.X.shape[0])))
            else:
                indices = rowitem.indices
            if sort_i[i] is not None:
                indices = indices[sort_i[i]]

            labels = [str(i) for i in indices]

            labelslist = TextListWidget(
                items=labels, parent=widget, orientation=Qt.Vertical)

            labelslist._indices = indices
            labelslist.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            labelslist.setContentsMargins(0.0, 0.0, 0.0, 0.0)
            labelslist.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

            grid.addItem(labelslist, Row0 + i * 2 + 1, RightLabelColumn)
            grid.setAlignment(labelslist, Qt.AlignLeft)
            row_annotation_widgets.append(labelslist)

        for j, colitem in enumerate(parts.columns):
            # Top attr annotations
            if isinstance(colitem.indices, slice):
                indices = np.array(
                    range(*colitem.indices.indices(data.X.shape[1])))
            else:
                indices = colitem.indices
            if sort_j[j] is not None:
                indices = indices[sort_j[j]]

            labels = [data.domain[i].name for i in indices]

            labelslist = TextListWidget(
                items=labels, parent=widget, orientation=Qt.Horizontal)
            labelslist.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            labelslist._indices = indices

            labelslist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            grid.addItem(labelslist, TopLabelsRow, Col0 + j,
                         Qt.AlignBottom | Qt.AlignLeft)
            col_annotation_widgets.append(labelslist)
            col_annotation_widgets_top.append(labelslist)

            # Bottom attr annotations
            labelslist = TextListWidget(
                items=labels, parent=widget, orientation=Qt.Horizontal)
            labelslist.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            labelslist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            grid.addItem(labelslist, BottomLabelsRow, Col0 + j)
            col_annotation_widgets.append(labelslist)
            col_annotation_widgets_bottom.append(labelslist)

        legend = GradientLegendWidget(
            parts.levels[0], parts.levels[1], self.threshold_low, self.threshold_high,
            parent=widget)

        legend.set_color_table(palette, self.center_palette)
        legend.setMinimumSize(QSizeF(100, 20))
        legend.setVisible(self.legend)

        grid.addItem(legend, LegendRow, Col0)

        self.heatmap_scene.widget = widget
        self.heatmap_widget_grid = heatmap_widgets
        self.selection_manager.set_heatmap_widgets(heatmap_widgets)

        self.row_annotation_widgets = row_annotation_widgets
        self.col_annotation_widgets = col_annotation_widgets
        self.col_annotation_widgets_top = col_annotation_widgets_top
        self.col_annotation_widgets_bottom = col_annotation_widgets_bottom
        self.col_dendrograms = column_dendrograms
        self.row_dendrograms = row_dendrograms

        self.update_annotations()
        self.update_column_annotations()

        self.__update_size_constraints()

    def __update_size_constraints(self):
        if self.heatmap_scene.widget is not None:
            mode = Qt.KeepAspectRatio if self.keep_aspect \
                   else Qt.IgnoreAspectRatio
            # get the preferred size from the view (view size - space for
            # scrollbars)
            view = self.sceneView
            size = view.size()
            fw = view.frameWidth()
            vsb_extent = view.verticalScrollBar().sizeHint().width()
            hsb_extent = view.horizontalScrollBar().sizeHint().height()
            size = QSizeF(max(size.width() - 2 * fw - vsb_extent, 0),
                          max(size.height() - 2 * fw - hsb_extent, 0))
            widget = self.heatmap_scene.widget
            layout = widget.layout()
            if mode == Qt.IgnoreAspectRatio:
                # Reset the row height constraints ...
                for i, hm_row in enumerate(self.heatmap_widget_grid):
                    layout.setRowMaximumHeight(3 + i * 2 + 1, np.finfo(np.float32).max)
                    layout.setRowPreferredHeight(3 + i * 2 + 1, 0)
                # ... and resize to match the viewport, taking the minimum size
                # into account
                minsize = widget.minimumSize()
                size = size.expandedTo(minsize)
                preferred = widget.effectiveSizeHint(Qt.PreferredSize)
                widget.resize(preferred.boundedTo(size))
            else:
                # First set/update the widget's width (the layout will
                # distribute the available width to heatmap widgets in
                # the grid)
                minsize = widget.minimumSize()
                preferred = widget.effectiveSizeHint(Qt.PreferredSize)

                if preferred.width() < size.expandedTo(minsize).width():
                    size = preferred

                widget.resize(size.expandedTo(minsize).width(),
                              widget.size().height())
                # calculate and set the heatmap row's heights based on
                # the width
                for i, hm_row in enumerate(self.heatmap_widget_grid):
                    heights = []
                    for hm in hm_row:
                        hm_size = QSizeF(hm.heatmap_item.pixmap().size())
                        hm_size = scaled(
                            hm_size, QSizeF(hm.size().width(), -1),
                            Qt.KeepAspectRatioByExpanding)

                        heights.append(hm_size.height())
                    layout.setRowMaximumHeight(3 + i * 2 + 1, max(heights))
                    layout.setRowPreferredHeight(3 + i * 2 + 1, max(heights))

                # set/update the widget's height
                constraint = QSizeF(size.width(), -1)
                sh = widget.effectiveSizeHint(Qt.PreferredSize, constraint)
                minsize = widget.effectiveSizeHint(Qt.MinimumSize, constraint)
                sh = sh.expandedTo(minsize).expandedTo(widget.minimumSize())

#                 print("Resize 2", sh)
#                 print("  old:", widget.size().width(), widget.size().height())
#                 print("  new:", widget.size().width(), sh.height())

                widget.resize(sh)
#                 print("Did resize")
            self.__fixup_grid_layout()

    def __fixup_grid_layout(self):
        self.__update_margins()
        self.__update_scene_rects()
        self.__update_selection_geometry()

    def __update_scene_rects(self):
        rect = self.scene.widget.geometry()
        self.heatmap_scene.setSceneRect(rect)

        spacing = self.scene.widget.layout().rowSpacing(2)
        headerrect = QRectF(rect)
        headerrect.setBottom(
            max((w.geometry().bottom()
                 for w in (self.col_annotation_widgets_top +
                           self.col_dendrograms)
                 if w is not None and w.isVisible()),
                default=rect.top())
        )

        if not headerrect.isEmpty():
            headerrect = headerrect.adjusted(0, 0, 0, spacing / 2)

        footerrect = QRectF(rect)
        footerrect.setTop(
            min((w.geometry().top() for w in self.col_annotation_widgets_bottom
                 if w is not None and w.isVisible()),
                default=rect.bottom())
        )
        if not footerrect.isEmpty():
            footerrect = footerrect.adjusted(0, - spacing / 2, 0, 0)

        self.sceneView.setSceneRect(rect)
        self.sceneView.setHeaderSceneRect(headerrect)
        self.sceneView.setFooterSceneRect(footerrect)

    def __on_layout_activate(self):
        self.__update_scene_rects()
        self.__update_selection_geometry()

    def __aspect_mode_changed(self):
        self.__update_size_constraints()

    def eventFilter(self, reciever, event):
        if reciever is self.sceneView.viewport() and \
                event.type() == QEvent.Resize:
            self.__update_size_constraints()

        return super().eventFilter(reciever, event)

    def __update_margins(self):
        """
        Update horizontal dendrogram and text list widgets margins to
        include the space for average stripe.
        """
        def offset(hm):
            if hm.show_averages:
                return hm.averages_item.size().width()
            else:
                return 0

        hm_row = self.heatmap_widget_grid[0]
        dendrogram_col = self.col_dendrograms

        col_annot = zip(self.col_annotation_widgets_top,
                        self.col_annotation_widgets_bottom)

        for hm, annot, dendrogram in zip(hm_row, col_annot, dendrogram_col):
            left_offset = offset(hm)
            if dendrogram is not None:
                _, top, right, bottom = dendrogram.getContentsMargins()
                dendrogram.setContentsMargins(
                    left_offset, top, right, bottom)

            _, top, right, bottom = annot[0].getContentsMargins()
            annot[0].setContentsMargins(left_offset, top, right, bottom)
            _, top, right, bottom = annot[1].getContentsMargins()
            annot[1].setContentsMargins(left_offset, top, right, bottom)

    def __update_clustering_enable_state(self, data):
        if data is not None:
            N = len(data)
            M = len(data.domain.attributes)
        else:
            N = M = 0

        rc_enabled = N <= self.MaxClustering
        rco_enabled = N <= self.MaxOrderedClustering
        cc_enabled = M <= self.MaxClustering
        cco_enabled = M <= self.MaxOrderedClustering
        row_clust, col_clust = self.row_clustering, self.col_clustering

        row_clust_msg = ""
        col_clust_msg = ""

        if not rco_enabled and row_clust == Clustering.OrderedClustering:
            row_clust = Clustering.Clustering
            row_clust_msg = "Row cluster ordering was disabled due to the " \
                            "input matrix being to big"
        if not rc_enabled and row_clust == Clustering.Clustering:
            row_clust = Clustering.None_
            row_clust_msg = "Row clustering was was disabled due to the " \
                            "input matrix being to big"

        if not cco_enabled and col_clust == Clustering.OrderedClustering:
            col_clust = Clustering.Clustering
            col_clust_msg = "Column cluster ordering was disabled due to " \
                            "the input matrix being to big"
        if not cc_enabled and col_clust == Clustering.Clustering:
            col_clust = Clustering.None_
            col_clust_msg = "Column clustering was disabled due to the " \
                            "input matrix being to big"

        self.col_clustering = col_clust
        self.row_clustering = row_clust

        self.Information.row_clust(row_clust_msg, shown=bool(row_clust_msg))
        self.Information.col_clust(col_clust_msg, shown=bool(col_clust_msg))

        # Disable/enable the combobox items for the clustering methods
        def setenabled(cb: QComboBox, clu: bool, clu_op: bool):
            model = cb.model()
            assert isinstance(model, QStandardItemModel)
            idx = cb.findData(Clustering.OrderedClustering, ClusteringRole)
            assert idx != -1
            model.item(idx).setEnabled(clu_op)
            idx = cb.findData(Clustering.Clustering, ClusteringRole)
            assert idx != -1
            model.item(idx).setEnabled(clu)

        setenabled(self.row_cluster_cb, rc_enabled, rco_enabled)
        setenabled(self.col_cluster_cb, cc_enabled, cco_enabled)

    def heatmap_widgets(self):
        """Iterate over heatmap widgets.
        """
        for item in self.heatmap_scene.items():
            if isinstance(item, GraphicsHeatmapWidget):
                yield item

    def label_widgets(self):
        """Iterate over GraphicsSimpleTextList widgets.
        """
        for item in self.heatmap_scene.items():
            if isinstance(item, TextListWidget):
                yield item

    def dendrogram_widgets(self):
        """Iterate over dendrogram widgets
        """
        for item in self.heatmap_scene.items():
            if isinstance(item, DendrogramWidget):
                yield item

    def legend_widgets(self):
        for item in self.heatmap_scene.items():
            if isinstance(item, GradientLegendWidget):
                yield item

    def update_averages_stripe(self):
        """Update the visibility of the averages stripe.
        """
        if self.effective_data is not None:
            for widget in self.heatmap_widgets():
                widget.set_show_averages(self.averages)
                widget.layout().activate()

            self.scene.widget.layout().activate()
            self.__fixup_grid_layout()

    def update_grid_spacing(self):
        """Update layout spacing.
        """
        if self.scene.widget:
            layout = self.scene.widget.layout()
            layout.setSpacing(self.space_x)
            self.__fixup_grid_layout()

    def update_lowslider(self):
        low, high = self.controls.threshold_low, self.controls.threshold_high
        if low.value() >= high.value():
            low.setSliderPosition(high.value() - 1)
        self.update_color_schema()

    def update_highslider(self):
        low, high = self.controls.threshold_low, self.controls.threshold_high
        if low.value() >= high.value():
            high.setSliderPosition(low.value() + 1)
        self.update_color_schema()

    def update_color_schema(self):
        self.palette_name = self.color_cb.currentData().name
        palette = self.color_palette()
        for heatmap in self.heatmap_widgets():
            heatmap.set_thresholds(self.threshold_low, self.threshold_high)
            heatmap.set_color_table(palette, self.center_palette)

        for legend in self.legend_widgets():
            legend.set_thresholds(self.threshold_low, self.threshold_high)
            legend.set_color_table(palette, self.center_palette)

    def __update_column_clustering(self):
        self.update_heatmaps()
        self.commit()

    def __update_row_clustering(self):
        self.update_heatmaps()
        self.commit()

    def update_legend(self):
        for item in self.heatmap_scene.items():
            if isinstance(item, GradientLegendWidget):
                item.setVisible(self.legend)

    def update_annotations(self):
        if self.input_data is not None:
            var = self.annotation_var
            show = var is not None
            if show:
                annot_col, _ = self.input_data.get_column_view(var)
            else:
                annot_col = None

            if self.merge_kmeans and self.kmeans_model is not None:
                merge_indices = self.merge_indices
            else:
                merge_indices = None

            for labelslist in self.row_annotation_widgets:
                labelslist.setVisible(bool(show))
                if show:
                    indices = labelslist._indices
                    if merge_indices is not None:
                        join = lambda values: (
                            join_ellided(", ", 42, values, " ({} more)")
                        )
                        # collect all original labels for every merged row
                        values = [annot_col[merge_indices[i]] for i in indices]
                        labels = [join(list(map(var.str_val, vals)))
                                  for vals in values]
                    else:
                        data = annot_col[indices]
                        labels = [var.str_val(val) for val in data]

                    labelslist.setItems(labels)

    def update_column_annotations(self):
        if self.data is not None:
            show_top = self.column_label_pos & OWHeatMap.PositionTop
            show_bottom = self.column_label_pos & OWHeatMap.PositionBottom

            for labelslist in self.col_annotation_widgets_top:
                labelslist.setVisible(show_top)

            TopLabelsRow = 2
            Row0 = 3
            BottomLabelsRow = Row0 + 2 * len(self.heatmapparts.rows)

            layout = self.heatmap_scene.widget.layout()
            layout.setRowMaximumHeight(TopLabelsRow, -1 if show_top else 0)
            layout.setRowSpacing(TopLabelsRow, -1 if show_top else 0)

            for labelslist in self.col_annotation_widgets_bottom:
                labelslist.setVisible(show_bottom)

            layout.setRowMaximumHeight(BottomLabelsRow, -1 if show_top else 0)

            self.__fixup_grid_layout()

    def __select_by_cluster(self, item, dendrogramindex):
        # User clicked on a dendrogram node.
        # Select all rows corresponding to the cluster item.
        node = item.node
        try:
            hm = self.heatmap_widget_grid[dendrogramindex][0]
        except IndexError:
            pass
        else:
            key = QApplication.keyboardModifiers()
            clear = not (key & ((Qt.ControlModifier | Qt.ShiftModifier |
                                 Qt.AltModifier)))
            remove = (key & (Qt.ControlModifier | Qt.AltModifier))
            append = (key & Qt.ControlModifier)
            self.selection_manager.selection_add(
                node.value.first, node.value.last - 1, hm,
                clear=clear, remove=remove, append=append)

    def __update_selection_geometry(self):
        for item in self.selection_rects:
            item.setParentItem(None)
            self.heatmap_scene.removeItem(item)

        self.selection_rects = []
        self.selection_manager.update_selection_rects()
        rects = self.selection_manager.selection_rects
        for rect in rects:
            item = QGraphicsRectItem(rect, None)
            pen = QPen(Qt.black, 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            self.heatmap_scene.addItem(item)
            self.selection_rects.append(item)

    def on_selection_finished(self):
        self.selected_rows = self.selection_manager.selections
        self.commit()

    def commit(self):
        data = None
        indices = None
        if self.merge_kmeans:
            merge_indices = self.merge_indices
        else:
            merge_indices = None

        if self.input_data is not None and self.selected_rows:
            sortind = np.hstack([labels._indices
                                 for labels in self.row_annotation_widgets])
            indices = sortind[self.selected_rows]

            if merge_indices is not None:
                # expand merged indices
                indices = np.hstack([merge_indices[i] for i in indices])

            data = self.input_data[indices]

        self.Outputs.selected_data.send(data)
        self.Outputs.annotated_data.send(create_annotated_table(self.input_data, indices))

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()

    def send_report(self):
        self.report_items((
            ("Columns:", "Clustering" if self.col_clustering else "No sorting"),
            ("Rows:", "Clustering" if self.row_clustering else "No sorting"),
            ("Split:",
             self.split_by_var is not None and self.split_by_var.name),
            ("Row annotation",
             self.annotation_var is not None and self.annotation_var.name),
        ))
        self.report_plot()

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is not None and version < 3:
            def st2cl(state: bool) -> Clustering:
                return Clustering.OrderedClustering if state else \
                       Clustering.None_
            rc = settings.pop("row_clustering", False)
            cc = settings.pop("col_clustering", False)
            settings["row_clustering_method"] = st2cl(rc).name
            settings["col_clustering_method"] = st2cl(cc).name


class GraphicsWidget(QGraphicsWidget):
    """A graphics widget which can notify on relayout events.
    """
    #: The widget's layout has activated (i.e. did a relayout
    #: of the widget's contents)
    layoutDidActivate = Signal()

    def event(self, event):
        rval = super().event(event)
        if event.type() == QEvent.LayoutRequest and self.layout() is not None:
            self.layoutDidActivate.emit()
        return rval


class GraphicsPixmapWidget(QGraphicsWidget):
    def __init__(self, parent=None, pixmap=None, scaleContents=False,
                 aspectMode=Qt.KeepAspectRatio, **kwargs):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.__scaleContents = scaleContents
        self.__aspectMode = aspectMode

        self.__pixmap = pixmap or QPixmap()
        self.__item = QGraphicsPixmapItem(self.__pixmap, self)
        self.__updateScale()

    def setPixmap(self, pixmap):
        self.prepareGeometryChange()
        self.__pixmap = pixmap or QPixmap()
        self.__item.setPixmap(self.__pixmap)
        self.updateGeometry()

    def pixmap(self):
        return self.__pixmap

    def setAspectRatioMode(self, mode):
        if self.__aspectMode != mode:
            self.__aspectMode = mode

    def aspectRatioMode(self):
        return self.__aspectMode

    def setScaleContents(self, scale):
        if self.__scaleContents != scale:
            self.__scaleContents = bool(scale)
            self.updateGeometry()
            self.__updateScale()

    def scaleContents(self):
        return self.__scaleContents

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            sh = QSizeF(self.__pixmap.size())
            if self.__scaleContents:
                sh = scaled(sh, constraint, self.__aspectMode)
            return sh
        elif which == Qt.MinimumSize:
            if self.__scaleContents:
                return QSizeF(0, 0)
            else:
                return QSizeF(self.__pixmap.size())
        elif which == Qt.MaximumSize:
            if self.__scaleContents:
                return QSizeF()
            else:
                return QSizeF(self.__pixmap.size())
        else:
            # Qt.MinimumDescent
            return QSizeF()

    def setGeometry(self, rect):
        super().setGeometry(rect)
        crect = self.contentsRect()
        self.__item.setPos(crect.topLeft())
        self.__updateScale()

    def __updateScale(self):
        if self.__pixmap.isNull():
            return
        pxsize = QSizeF(self.__pixmap.size())
        crect = self.contentsRect()
        self.__item.setPos(crect.topLeft())

        if self.__scaleContents:
            csize = scaled(pxsize, crect.size(), self.__aspectMode)
        else:
            csize = pxsize

        xscale = csize.width() / pxsize.width()
        yscale = csize.height() / pxsize.height()

        t = QTransform().scale(xscale, yscale)
        self.__item.setTransform(t)

    def pixmapTransform(self):
        return QTransform(self.__item.transform())


class GraphicsHeatmapWidget(QGraphicsWidget):
    def __init__(self, parent=None, data=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setAcceptHoverEvents(True)

        self.__levels = None
        self.__threshold_low, self.__threshold_high = 0., 1.
        self.__center_palette = False
        self.__colortable = None
        self.__data = data

        self.__pixmap = QPixmap()
        self.__avgpixmap = QPixmap()

        layout = QGraphicsLinearLayout(Qt.Horizontal)
        layout.setContentsMargins(0, 0, 0, 0)
        self.heatmap_item = GraphicsPixmapWidget(
            self, scaleContents=True, aspectMode=Qt.IgnoreAspectRatio)

        self.averages_item = GraphicsPixmapWidget(
            self, scaleContents=True, aspectMode=Qt.IgnoreAspectRatio)

        layout.addItem(self.averages_item)
        layout.addItem(self.heatmap_item)
        layout.setItemSpacing(0, 2)

        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.show_averages = True

        self.set_heatmap_data(data)

    def clear(self):
        """Clear/reset the widget."""
        self.__data = None
        self.__pixmap = None
        self.__avgpixmap = None

        self.heatmap_item.setPixmap(QPixmap())
        self.averages_item.setPixmap(QPixmap())
        self.show_averages = True
        self.updateGeometry()
        self.layout().invalidate()

    def set_heatmap(self, heatmap):
        """Set the heatmap data for display.
        """
        self.clear()

        self.set_heatmap_data(heatmap)
        self.update()

    def set_heatmap_data(self, data):
        """Set the heatmap data for display."""
        if self.__data is not data:
            self.clear()
            self.__data = data
            self._update_pixmap()
            self.update()

    def heatmap_data(self):
        if self.__data is not None:
            v = self.__data.view()
            v.flags.writeable = False
            return v
        else:
            return None

    def set_levels(self, levels):
        if levels != self.__levels:
            self.__levels = levels
            self._update_pixmap()
            self.update()

    def set_show_averages(self, show):
        if self.show_averages != show:
            self.show_averages = show
            self.averages_item.setVisible(show)
            self.averages_item.setMaximumWidth(-1 if show else 0)
            self.layout().invalidate()
            self.update()

    def set_color_table(self, table, center):
        self.__colortable = table
        self.__center_palette = center
        self._update_pixmap()
        self.update()

    def set_thresholds(self, threshold_low, threshold_high):
        self.__threshold_low = threshold_low
        self.__threshold_high = threshold_high
        self._update_pixmap()
        self.update()

    def _update_pixmap(self):
        """
        Update the pixmap if its construction arguments changed.
        """
        if self.__data is not None:
            if self.__colortable is not None:
                lut = self.__colortable
            else:
                lut = None

            ll, lh = self.__levels
            ll, lh = levels_with_thresholds(ll, lh, self.__threshold_low, self.__threshold_high,
                                            self.__center_palette)

            argb, _ = pg.makeARGB(
                self.__data, lut=lut, levels=(ll, lh))
            argb[np.isnan(self.__data)] = (100, 100, 100, 255)

            qimage = pg.makeQImage(argb, transpose=False)
            self.__pixmap = QPixmap.fromImage(qimage)
            avg = np.nanmean(self.__data, axis=1, keepdims=True)
            argb, _ = pg.makeARGB(
                avg, lut=lut, levels=(ll, lh))
            qimage = pg.makeQImage(argb, transpose=False)
            self.__avgpixmap = QPixmap.fromImage(qimage)
        else:
            self.__pixmap = QPixmap()
            self.__avgpixmap = QPixmap()

        self.heatmap_item.setPixmap(self.__pixmap)
        self.averages_item.setPixmap(self.__avgpixmap)
        hmsize = QSizeF(self.__pixmap.size())
        avsize = QSizeF(self.__avgpixmap.size())

        self.heatmap_item.setMinimumSize(hmsize)
        self.averages_item.setMinimumSize(avsize)
        self.heatmap_item.setPreferredSize(hmsize * 10)
        self.averages_item.setPreferredSize(avsize * 10)
        self.layout().invalidate()

    def cell_at(self, pos):
        """Return the cell row, column from `pos` in local coordinates.
        """
        if self.__pixmap.isNull() or not (
                self.heatmap_item.geometry().contains(pos) or
                self.averages_item.geometry().contains(pos)):
            return (-1, -1)

        if self.heatmap_item.geometry().contains(pos):
            item_clicked = self.heatmap_item
        elif self.averages_item.geometry().contains(pos):
            item_clicked = self.averages_item
        pos = self.mapToItem(item_clicked, pos)
        size = self.heatmap_item.size()

        x, y = pos.x(), pos.y()

        N, M = self.__data.shape
        fx = x / size.width()
        fy = y / size.height()
        i = min(int(math.floor(fy * N)), N - 1)
        j = min(int(math.floor(fx * M)), M - 1)
        return i, j

    def cell_rect(self, row, column):
        """Return a rectangle in local coordinates containing the cell
        at `row` and `column`.
        """
        size = self.__pixmap.size()
        if not (0 <= column < size.width() or 0 <= row < size.height()):
            return QRectF()

        topleft = QPointF(column, row)
        bottomright = QPointF(column + 1, row + 1)
        t = self.heatmap_item.pixmapTransform()
        rect = t.mapRect(QRectF(topleft, bottomright))
        rect.translated(self.heatmap_item.pos())
        return rect

    def row_rect(self, row):
        """
        Return a QRectF in local coordinates containing the entire row.
        """
        rect = self.cell_rect(row, 0)
        rect.setLeft(0)
        rect.setRight(self.size().width())
        return rect

    def cell_tool_tip(self, row, column):
        return "{}, {}: {:g}".format(row, column, self.__data[row, column])

    def hoverMoveEvent(self, event):
        pos = event.pos()
        row, column = self.cell_at(pos)
        if row != -1:
            tooltip = self.cell_tool_tip(row, column)
            # TODO: Move/delegate to (Scene) helpEvent
            self.setToolTip(tooltip)
        return super().hoverMoveEvent(event)


class HeatmapScene(QGraphicsScene):
    """A Graphics Scene with heatmap widgets."""
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.selection_manager = HeatmapSelectionManager()
        self.__selecting = False

    def set_selection_manager(self, manager):
        self.selection_manager = manager

    def _items(self, pos=None, cls=object):
        if pos is not None:
            items = self.items(QRectF(pos, QSizeF(3, 3)).translated(-1.5, -1.5))
        else:
            items = self.items()

        for item in items:
            if isinstance(item, cls):
                yield item

    def heatmap_at_pos(self, pos):
        items = list(self._items(pos, GraphicsHeatmapWidget))
        if items:
            return items[0]
        else:
            return None

    def heatmap_widgets(self):
        return self._items(None, GraphicsHeatmapWidget)

    def select_from_dendrogram(self, dendrogram, key):
        """Select all heatmap rows which belong to the dendrogram.
        """
        dendrogram_widget = dendrogram.parentWidget()
        anchors = list(dendrogram_widget.leaf_anchors())
        cluster = dendrogram.cluster
        start, end = anchors[cluster.first], anchors[cluster.last - 1]
        start, end = dendrogram_widget.mapToScene(start), dendrogram_widget.mapToScene(end)
        # Find a heatmap widget containing start and end y coordinates.

        heatmap = None
        for hm in self.heatmap_widgets():
            b_rect = hm.sceneBoundingRect()
            if b_rect.contains(QPointF(b_rect.center().x(), start.y())):
                heatmap = hm
                break

        if dendrogram:
            b_rect = heatmap.boundingRect()
            start, end = heatmap.mapFromScene(start), heatmap.mapFromScene(end)
            start, _ = heatmap.cell_at(QPointF(b_rect.center().x(), start.y()))
            end, _ = heatmap.cell_at(QPointF(b_rect.center().x(), end.y()))
            clear = not (key & ((Qt.ControlModifier | Qt.ShiftModifier |
                                 Qt.AltModifier)))
            remove = (key & (Qt.ControlModifier | Qt.AltModifier))
            append = (key & Qt.ControlModifier)
            self.selection_manager.selection_add(
                start, end, heatmap, clear=clear, remove=remove, append=append)

    def mousePressEvent(self, event):
        pos = event.scenePos()
        heatmap = self.heatmap_at_pos(pos)
        if heatmap and event.button() & Qt.LeftButton:
            row, _ = heatmap.cell_at(heatmap.mapFromScene(pos))
            if row != -1:
                self.selection_manager.selection_start(heatmap, event)
                self.__selecting = True
        return QGraphicsScene.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        pos = event.scenePos()
        heatmap = self.heatmap_at_pos(pos)
        if heatmap and event.buttons() & Qt.LeftButton and self.__selecting:
            row, _ = heatmap.cell_at(heatmap.mapFromScene(pos))
            if row != -1:
                self.selection_manager.selection_update(heatmap, event)
        return QGraphicsScene.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        pos = event.scenePos()
        heatmap = self.heatmap_at_pos(pos)
        if heatmap and event.button() == Qt.LeftButton and self.__selecting:
            self.selection_manager.selection_finish(heatmap, event)

        if event.button() == Qt.LeftButton and self.__selecting:
            self.__selecting = False

        return QGraphicsScene.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        return QGraphicsScene.mouseDoubleClickEvent(self, event)


class GraphicsSimpleTextLayoutItem(QGraphicsLayoutItem):
    """ A Graphics layout item wrapping a QGraphicsSimpleTextItem alowing it
    to be managed by a layout.

    """
    def __init__(self, text_item, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.text_item = text_item
        if orientation == Qt.Vertical:
            self.text_item.rotate(-90)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        if self.orientation == Qt.Horizontal:
            self.text_item.setPos(rect.topLeft())
        else:
            self.text_item.setPos(rect.bottomLeft())

    def sizeHint(self, which, constraint=QSizeF()):
        if which in [Qt.PreferredSize]:
            size = self.text_item.boundingRect().size()
            if self.orientation == Qt.Horizontal:
                return size
            else:
                return QSizeF(size.height(), size.width())
        else:
            return QSizeF()

    def updateGeometry(self):
        super().updateGeometry()
        parent = self.parentLayoutItem()
        if parent.isLayout():
            parent.updateGeometry()

    def setFont(self, font):
        self.text_item.setFont(font)
        self.updateGeometry()

    def setText(self, text):
        self.text_item.setText(text)
        self.updateGeometry()


class GradientLegendWidget(QGraphicsWidget):
    def __init__(self, low, high, threshold_low, threshold_high, parent=None):
        super().__init__(parent)
        self.low = low
        self.high = high
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.center_palette = False
        self.color_table = None

        layout = QGraphicsLinearLayout(Qt.Vertical)
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        layout_labels = QGraphicsLinearLayout(Qt.Horizontal)
        layout.addItem(layout_labels)
        layout_labels.setContentsMargins(0, 0, 0, 0)
        label_lo = QGraphicsSimpleTextItem("%.2f" % low, self)
        label_hi = QGraphicsSimpleTextItem("%.2f" % high, self)
        self.item_low = GraphicsSimpleTextLayoutItem(label_lo, parent=self)
        self.item_high = GraphicsSimpleTextLayoutItem(label_hi, parent=self)

        layout_labels.addItem(self.item_low)
        layout_labels.addStretch(10)
        layout_labels.addItem(self.item_high)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.__pixitem = GraphicsPixmapWidget(parent=self, scaleContents=True,
                                              aspectMode=Qt.IgnoreAspectRatio)
        self.__pixitem.setMinimumHeight(12)
        layout.addItem(self.__pixitem)
        self.__update()

    def set_color_table(self, color_table, center):
        self.color_table = color_table
        self.center_palette = center
        self.__update()

    def set_thresholds(self, threshold_low, threshold_high):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.__update()

    def __update(self):
        data = np.linspace(self.low, self.high, num=1000)
        data = data.reshape((1, -1))
        ll, lh = levels_with_thresholds(self.low, self.high,
                                        self.threshold_low, self.threshold_high,
                                        self.center_palette)
        argb, _ = pg.makeARGB(data, lut=self.color_table,
                              levels=(ll, lh))
        qimg = pg.makeQImage(argb, transpose=False)
        self.__pixitem.setPixmap(QPixmap.fromImage(qimg))

        self.item_low.setText("%.2f" % self.low)
        self.item_high.setText("%.2f" % self.high)
        self.layout().invalidate()


class HeatmapSelectionManager(QObject):
    """Selection manager for heatmap rows
    """
    selection_changed = Signal()
    selection_finished = Signal()

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.selections = []
        self.selection_ranges = []
        self.selection_ranges_temp = []
        self.heatmap_widgets = []
        self.selection_rects = []
        self.heatmaps = []
        self._heatmap_ranges = {}
        self._start_row = 0

    def clear(self):
        self.remove_rows(self.selection)

    def set_heatmap_widgets(self, widgets):
        self.remove_rows(self.selections)
        self.heatmaps = list(zip(*widgets))

        # Compute row ranges for all heatmaps
        self._heatmap_ranges = {}
        start = end = 0

        for group in zip(*widgets):
            start = end = 0
            for heatmap in group:
                end += heatmap.heatmap_data().shape[0]
                self._heatmap_ranges[heatmap] = (start, end)
                start = end

    def select_rows(self, rows, heatmap=None, clear=True):
        """Add `rows` to selection. If `heatmap` is provided the rows
        are mapped from the local indices to global heatmap indices. If `clear`
        then remove previous rows.
        """
        if heatmap is not None:
            start, _ = self._heatmap_ranges[heatmap]
            rows = [start + r for r in rows]

        old_selection = list(self.selections)
        if clear:
            self.selections = rows
        else:
            self.selections = sorted(set(self.selections + rows))

        if self.selections != old_selection:
            self.update_selection_rects()
            self.selection_changed.emit()

    def remove_rows(self, rows):
        """Remove `rows` from the selection.
        """
        old_selection = list(self.selections)
        self.selections = sorted(set(self.selections) - set(rows))
        if old_selection != self.selections:
            self.update_selection_rects()
            self.selection_changed.emit()

    def combined_ranges(self, ranges):
        combined_ranges = set()
        for start, end in ranges:
            if start <= end:
                rng = range(start, end + 1)
            else:
                rng = range(start, end - 1, -1)
            combined_ranges.update(rng)
        return sorted(combined_ranges)

    def selection_start(self, heatmap_widget, event):
        """ Selection  started by `heatmap_widget` due to `event`.
        """
        pos = heatmap_widget.mapFromScene(event.scenePos())
        row, _ = heatmap_widget.cell_at(pos)

        start, _ = self._heatmap_ranges[heatmap_widget]
        row = start + row
        self._start_row = row
        range = (row, row)
        self.selection_ranges_temp = []
        if event.modifiers() & Qt.ControlModifier:
            self.selection_ranges_temp = self.selection_ranges
            self.selection_ranges = self.remove_range(
                self.selection_ranges, row, row, append=True)
        elif event.modifiers() & Qt.ShiftModifier:
            self.selection_ranges.append(range)
        elif event.modifiers() & Qt.AltModifier:
            self.selection_ranges = self.remove_range(
                self.selection_ranges, row, row, append=False)
        else:
            self.selection_ranges = [range]
        self.select_rows(self.combined_ranges(self.selection_ranges))

    def selection_update(self, heatmap_widget, event):
        """ Selection updated by `heatmap_widget due to `event` (mouse drag).
        """
        pos = heatmap_widget.mapFromScene(event.scenePos())
        row, _ = heatmap_widget.cell_at(pos)
        if row < 0:
            return

        start, _ = self._heatmap_ranges[heatmap_widget]
        row = start + row
        if event.modifiers() & Qt.ControlModifier:
            self.selection_ranges = self.remove_range(
                self.selection_ranges_temp, self._start_row, row, append=True)
        elif event.modifiers() & Qt.AltModifier:
            self.selection_ranges = self.remove_range(
                self.selection_ranges, self._start_row, row, append=False)
        else:
            if self.selection_ranges:
                self.selection_ranges[-1] = (self._start_row, row)
            else:
                self.selection_ranges = [(row, row)]

        self.select_rows(self.combined_ranges(self.selection_ranges))

    def selection_finish(self, heatmap_widget, event):
        """ Selection finished by `heatmap_widget due to `event`.
        """
        pos = heatmap_widget.mapFromScene(event.scenePos())
        row, _ = heatmap_widget.cell_at(pos)
        start, _ = self._heatmap_ranges[heatmap_widget]
        row = start + row
        if event.modifiers() & Qt.ControlModifier:
            pass
        elif event.modifiers() & Qt.AltModifier:
            self.selection_ranges = self.remove_range(
                self.selection_ranges, self._start_row, row, append=False)
        else:
            if len(self.selection_ranges) > 0:
                self.selection_ranges[-1] = (self._start_row, row)
        self.select_rows(self.combined_ranges(self.selection_ranges))
        self.selection_finished.emit()

    def selection_add(self, start, end, heatmap=None, clear=True,
                      remove=False, append=False):
        """ Add/remove a selection range from `start` to `end`.
        """
        if heatmap is not None:
            _start, _ = self._heatmap_ranges[heatmap]
            start = _start + start
            end = _start + end

        if clear:
            self.selection_ranges = []
        if remove:
            self.selection_ranges = self.remove_range(
                self.selection_ranges, start, end, append=append)
        else:
            self.selection_ranges.append((start, end))
        self.select_rows(self.combined_ranges(self.selection_ranges))
        self.selection_finished.emit()

    def remove_range(self, ranges, start, end, append=False):
        if start > end:
            start, end = end, start
        comb_ranges = [i for i in self.combined_ranges(ranges)
                       if i > end or i < start]
        if append:
            comb_ranges += [i for i in range(start, end + 1)
                            if i not in self.combined_ranges(ranges)]
            comb_ranges = sorted(comb_ranges)
        return self.combined_to_ranges(comb_ranges)

    def combined_to_ranges(self, comb_ranges):
        ranges = []
        if len(comb_ranges) > 0:
            i, start, end = 0, comb_ranges[0], comb_ranges[0]
            for val in comb_ranges[1:]:
                i += 1
                if start + i < val:
                    ranges.append((start, end))
                    i, start = 0, val
                end = val
            ranges.append((start, end))
        return ranges

    def update_selection_rects(self):
        """ Update the selection rects.
        """
        def group_selections(selections):
            """Group selections along with heatmaps.
            """
            rows2hm = self.rows_to_heatmaps()
            selections = iter(selections)
            try:
                start = end = next(selections)
            except StopIteration:
                return
            end_heatmaps = rows2hm[end]
            try:
                while True:
                    new_end = next(selections)
                    new_end_heatmaps = rows2hm[new_end]
                    if new_end > end + 1 or new_end_heatmaps != end_heatmaps:
                        yield start, end, end_heatmaps
                        start = end = new_end
                        end_heatmaps = new_end_heatmaps
                    else:
                        end = new_end

            except StopIteration:
                yield start, end, end_heatmaps

        def selection_rect(start, end, heatmaps):
            rect = QRectF()
            for heatmap in heatmaps:
                h_start, _ = self._heatmap_ranges[heatmap]
                rect |= heatmap.mapToScene(heatmap.row_rect(start - h_start)).boundingRect()
                rect |= heatmap.mapToScene(heatmap.row_rect(end - h_start)).boundingRect()
            return rect

        self.selection_rects = []
        for start, end, heatmaps in group_selections(self.selections):
            rect = selection_rect(start, end, heatmaps)
            self.selection_rects.append(rect)

    def rows_to_heatmaps(self):
        heatmap_groups = zip(*self.heatmaps)
        rows2hm = {}
        for heatmaps in heatmap_groups:
            hm = heatmaps[0]
            start, end = self._heatmap_ranges[hm]
            rows2hm.update(dict.fromkeys(range(start, end), heatmaps))
        return rows2hm


def join_ellided(sep, maxlen, values, ellidetemplate="..."):
    def generate(sep, ellidetemplate, values):
        count = len(values)
        length = 0
        parts = []
        for i, val in enumerate(values):
            ellide = ellidetemplate.format(count - i) if count - i > 1 else ""
            parts.append(val)
            length += len(val) + (len(sep) if parts else 0)
            yield i, itertools.islice(parts, i + 1), length, ellide

    best = None
    for _, parts, length, ellide in generate(sep, ellidetemplate, values):
        if length > maxlen:
            if best is None:
                best = sep.join(parts) + ellide
            return best
        fulllen = length + len(ellide)
        if fulllen < maxlen or best is None:
            best = sep.join(parts) + ellide
    return best


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWHeatMap).run(Table("brown-selected.tab"))
