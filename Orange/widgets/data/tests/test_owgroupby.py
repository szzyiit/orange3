import os
import unittest
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from AnyQt import QtCore
from AnyQt.QtCore import QItemSelectionModel, Qt
from AnyQt.QtWidgets import QListView

from Orange.data import (
    Table,
    table_to_frame,
    Domain,
    ContinuousVariable,
)
from Orange.data.tests.test_aggregate import create_sample_data
from Orange.widgets.data.owgroupby import OWGroupBy
from Orange.widgets.tests.base import WidgetTest


class TestOWGroupBy(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWGroupBy)
        self.iris = Table("iris")

        self.data = create_sample_data()

    def test_none_data(self):
        self.send_signal(self.widget.Inputs.data, None)

        self.assertEqual(self.widget.agg_table_model.rowCount(), 0)
        self.assertEqual(self.widget.gb_attrs_model.rowCount(), 0)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_data(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        self.assertEqual(self.widget.agg_table_model.rowCount(), 5)
        self.assertEqual(self.widget.gb_attrs_model.rowCount(), 5)

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 35)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_data_domain_changed(self):
        self.send_signal(self.widget.Inputs.data, self.iris[:, -2:])
        self.assert_aggregations_equal(["平均值", "串接(Concatenate)"])

        self.send_signal(self.widget.Inputs.data, self.iris[:, -3:])
        self.assert_aggregations_equal(["平均值", "平均值", "串接(Concatenate)"])
        self.select_table_rows(self.widget.agg_table_view, [0])

    @staticmethod
    def _set_selection(view: QListView, indices: List[int]):
        view.clearSelection()
        sm = view.selectionModel()
        model = view.model()
        for ind in indices:
            sm.select(model.index(ind), QItemSelectionModel.Select)

    def test_groupby_attr_selection(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 35)

        # select iris attribute with index 0
        self._set_selection(self.widget.gb_attrs_view, [0])
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 3)

        # select iris attribute with index 0
        self._set_selection(self.widget.gb_attrs_view, [0, 1])
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 57)

    def assert_enabled_cbs(self, enabled_true):
        enabled_actual = set(
            name for name, cb in self.widget.agg_checkboxes.items() if cb.isEnabled()
        )
        self.assertSetEqual(enabled_true, enabled_actual)

    @staticmethod
    def select_table_rows(table, rows):
        table.clearSelection()
        indexes = [table.model().index(r, 0) for r in rows]
        mode = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        for i in indexes:
            table.selectionModel().select(i, mode)

    def test_attr_table_row_selection(self):
        self.send_signal(self.widget.Inputs.data, self.data)

        model = self.widget.agg_table_model
        table = self.widget.agg_table_view

        self.assertListEqual(
            ["a", "b", "cvar", "dvar", "svar"],
            [model.data(model.index(i, 0)) for i in range(model.rowCount())],
        )

        self.select_table_rows(table, [0])
        self.assert_enabled_cbs(
            {
                "平均值",
                "中位数",
                "取模",
                "标准差",
                "方差",
                "和",
                "最小值",
                "最大值",
                "非缺失数量",
                "数量",
                "串接(Concatenate)",
                "跨度",
                "首值",
                "末值",
                "随机值",
                "非缺失占比",
            }
        )
        self.select_table_rows(table, [0, 1])
        self.assert_enabled_cbs(
            {
                "平均值",
                "中位数",
                "取模",
                "标准差",
                "方差",
                "和",
                "最小值",
                "最大值",
                "非缺失数量",
                "数量",
                "串接(Concatenate)",
                "跨度",
                "首值",
                "末值",
                "随机值",
                "非缺失占比",
            }
        )
        self.select_table_rows(table, [2])
        self.assert_enabled_cbs(
            {
                "平均值",
                "中位数",
                "取模",
                "标准差",
                "方差",
                "和",
                "最小值",
                "最大值",
                "非缺失数量",
                "数量",
                "串接(Concatenate)",
                "跨度",
                "首值",
                "末值",
                "随机值",
                "非缺失占比",
            }
        )
        self.select_table_rows(table, [3])  # discrete variable
        self.assert_enabled_cbs(
            {
                "非缺失数量",
                "数量",
                "串接(Concatenate)",
                "首值",
                "末值",
                "随机值",
                "非缺失占比",
            }
        )
        self.select_table_rows(table, [4])  # string variable
        self.assert_enabled_cbs(
            {
                "非缺失数量",
                "数量",
                "串接(Concatenate)",
                "首值",
                "末值",
                "随机值",
                "非缺失占比",
            }
        )
        self.select_table_rows(table, [3, 4])  # string variable
        self.assert_enabled_cbs(
            {
                "非缺失数量",
                "数量",
                "串接(Concatenate)",
                "首值",
                "末值",
                "随机值",
                "非缺失占比",
            }
        )
        self.select_table_rows(table, [2, 3, 4])  # string variable
        self.assert_enabled_cbs(
            {
                "平均值",
                "中位数",
                "取模",
                "标准差",
                "方差",
                "和",
                "最小值",
                "最大值",
                "非缺失数量",
                "数量",
                "串接(Concatenate)",
                "跨度",
                "首值",
                "末值",
                "随机值",
                "非缺失占比",
            }
        )

    def assert_aggregations_equal(self, expected_text):
        model = self.widget.agg_table_model
        agg_text = [model.data(model.index(i, 1)) for i in range(model.rowCount())]
        self.assertListEqual(expected_text, agg_text)

    def test_aggregations_change(self):
        table = self.widget.agg_table_view
        d = self.data.domain

        self.send_signal(self.widget.Inputs.data, self.data)

        self.assert_aggregations_equal(
            ["平均值", "平均值", "平均值", "串接(Concatenate)", "串接(Concatenate)"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值"},
                d["b"]: {"平均值"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0])
        self.widget.agg_checkboxes["中位数"].click()
        self.assert_aggregations_equal(
            ["平均值, 中位数", "平均值", "平均值", "串接(Concatenate)", "串接(Concatenate)"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "中位数"},
                d["b"]: {"平均值"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 1])
        self.widget.agg_checkboxes["取模"].click()
        self.assert_aggregations_equal(
            ["平均值, 中位数, 取模", "平均值, 取模", "平均值", "串接(Concatenate)", "串接(Concatenate)"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "中位数", "取模"},
                d["b"]: {"平均值", "取模"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 1])
        # median is partially checked and will become checked
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["中位数"].checkState()
        )
        self.widget.agg_checkboxes["中位数"].click()
        self.assertEqual(Qt.Checked, self.widget.agg_checkboxes["中位数"].checkState())
        self.assert_aggregations_equal(
            [
                "平均值, 中位数, 取模",
                "平均值, 中位数, 取模",
                "平均值",
                "串接(Concatenate)",
                "串接(Concatenate)",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "中位数", "取模"},
                d["b"]: {"平均值", "中位数", "取模"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["中位数"].click()
        self.assertEqual(
            Qt.Unchecked, self.widget.agg_checkboxes["中位数"].checkState()
        )
        self.assert_aggregations_equal(
            ["平均值, 取模", "平均值, 取模", "平均值", "串接(Concatenate)", "串接(Concatenate)"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "取模"},
                d["b"]: {"平均值", "取模"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 3])
        # median is unchecked and will change to partially checked
        self.assertEqual(
            Qt.Unchecked, self.widget.agg_checkboxes["中位数"].checkState()
        )
        self.widget.agg_checkboxes["中位数"].click()
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["中位数"].checkState()
        )
        self.assert_aggregations_equal(
            ["平均值, 中位数, 取模", "平均值, 取模", "平均值", "串接(Concatenate)", "串接(Concatenate)"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "中位数", "取模"},
                d["b"]: {"平均值", "取模"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["中位数"].click()
        self.assertEqual(
            Qt.Unchecked, self.widget.agg_checkboxes["中位数"].checkState()
        )
        self.assert_aggregations_equal(
            ["平均值, 取模", "平均值, 取模", "平均值", "串接(Concatenate)", "串接(Concatenate)"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "取模"},
                d["b"]: {"平均值", "取模"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["数量"].click()
        self.assertEqual(Qt.Checked, self.widget.agg_checkboxes["数量"].checkState())
        self.assert_aggregations_equal(
            [
                "平均值, 取模, 数量",
                "平均值, 取模",
                "平均值",
                "串接(Concatenate), 数量",
                "串接(Concatenate)",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "取模", "数量"},
                d["b"]: {"平均值", "取模"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"数量", "串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        # test the most complicated scenario: numeric with mode, numeric without
        # mode and discrete
        self.select_table_rows(table, [0])
        self.widget.agg_checkboxes["取模"].click()
        self.assert_aggregations_equal(
            ["平均值, 数量", "平均值, 取模", "平均值", "串接(Concatenate), 数量", "串接(Concatenate)"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "数量"},
                d["b"]: {"平均值", "取模"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"数量", "串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 1, 3])
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["取模"].checkState()
        )
        self.widget.agg_checkboxes["取模"].click()
        # must stay partially checked since one Continuous can still have mode
        # as a aggregation and discrete cannot have it
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["取模"].checkState()
        )
        self.assert_aggregations_equal(
            [
                "平均值, 取模, 数量",
                "平均值, 取模",
                "平均值",
                "串接(Concatenate), 数量",
                "串接(Concatenate)",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "取模", "数量"},
                d["b"]: {"平均值", "取模"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"数量", "串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        # since now all that can have Mode have it as an aggregation it can be
        # unchecked on the next click
        self.widget.agg_checkboxes["取模"].click()
        self.assertEqual(Qt.Unchecked, self.widget.agg_checkboxes["取模"].checkState())
        self.assert_aggregations_equal(
            ["平均值, 数量", "平均值", "平均值", "串接(Concatenate), 数量", "串接(Concatenate)"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "数量"},
                d["b"]: {"平均值"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"数量", "串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["取模"].click()
        self.widget.agg_checkboxes["非缺失数量"].click()
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["取模"].checkState()
        )
        self.assert_aggregations_equal(
            [
                "平均值, 取模, 非缺失数量 and 1 more",
                "平均值, 取模, 非缺失数量",
                "平均值",
                "串接(Concatenate), 非缺失数量, 数量",
                "串接(Concatenate)",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "取模", "数量", "非缺失数量"},
                d["b"]: {"平均值", "取模", "非缺失数量"},
                d["cvar"]: {"平均值"},
                d["dvar"]: {"数量", "非缺失数量", "串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

    def test_aggregation(self):
        """Test aggregation results"""
        self.send_signal(self.widget.Inputs.data, self.data)
        output = self.get_output(self.widget.Outputs.data)

        np.testing.assert_array_almost_equal(
            output.X, [[1, 2.143, 0.317], [2, 2, 2]], decimal=3
        )
        np.testing.assert_array_equal(
            output.metas,
            np.array(
                [
                    [
                        "val1 val2 val2 val1 val2 val1",
                        "sval1 sval2 sval2 sval1 sval2 sval1",
                        1.0,
                    ],
                    [
                        "val2 val1 val2 val1 val2 val1",
                        "sval2 sval1 sval2 sval1 sval2 sval1",
                        2.0,
                    ],
                ],
                dtype=object,
            ),
        )

        # select all aggregations for all features except a and b
        self._set_selection(self.widget.gb_attrs_view, [1, 2])
        self.select_table_rows(self.widget.agg_table_view, [2, 3, 4])
        # select all aggregations
        for cb in self.widget.agg_checkboxes.values():
            cb.click()
            while not cb.isChecked():
                cb.click()

        self.select_table_rows(self.widget.agg_table_view, [0, 1])
        # unselect all aggregations for attr a and b
        for cb in self.widget.agg_checkboxes.values():
            while cb.isChecked():
                cb.click()

        expected_columns = [
            "cvar - 平均值",
            "cvar - 中位数",
            "cvar - 取模",
            "cvar - 标准差",
            "cvar - 方差",
            "cvar - 和",
            "cvar - 最小值",
            "cvar - 最大值",
            "cvar - 跨度",
            "cvar - 首值",
            "cvar - 末值",
            "cvar - 非缺失数量",
            "cvar - 数量",
            "cvar - 非缺失占比",
            "dvar - 首值",
            "dvar - 末值",
            "dvar - 非缺失数量",
            "dvar - 数量",
            "dvar - 非缺失占比",
            "svar - 首值",
            "svar - 末值",
            "svar - 非缺失数量",
            "svar - 数量",
            "svar - 非缺失占比",
            "cvar - 串接(Concatenate)",
            "dvar - 串接(Concatenate)",
            "svar - 串接(Concatenate)",
            "a",  # groupby variables are last two in metas
            "b",
        ]

        # fmt: off
        expected_df = pd.DataFrame([
            [.15, .15, .1, .07, .005, .3, .1, .2, .1, 0.1, 0.2, 2, 2, 1,
             "val1", "val1", "val2", 2, 2, 1,
             "sval1", "sval2", 2, 2, 1,
             "0.1 0.2", "val1 val2", "sval1 sval2",
             1, 1],
            [.3, .3, .3, np.nan, np.nan, .3, .3, .3, 0, .3, .3, 1, 2, 0.5,
             "val2", "val2", "val2", 1, 2, 0.5,
             "", "sval2", 2, 2, 1,
             "0.3", "val2", "sval2",
             1, 2],
            [.433, .4, .3, 0.153, 0.023, 1.3, .3, .6, .3, .3, .6, 3, 3, 1,
             "val1", "val1", "val1", 3, 3, 1,
             "sval1", "sval1", 3, 3, 1,
             "0.3 0.4 0.6", "val1 val2 val1", "sval1 sval2 sval1",
             1, 3],
            [1.5, 1.5, 1, 0.707, 0.5, 3, 1, 2, 1, 1, 2, 2, 2, 1,
             "val1", "val2", "val1", 2, 2, 1,
             "sval2", "sval1", 2, 2, 1,
             "1.0 2.0", "val2 val1", "sval2 sval1",
             2, 1],
            [-0.5, -0.5, -4, 4.95, 24.5, -1, -4, 3, 7, 3, -4, 2, 2, 1,
             "val1", "val2", "val1", 2, 2, 1,
             "sval2", "sval1", 2, 2, 1,
             "3.0 -4.0", "val2 val1", "sval2 sval1",
             2, 2],
            [5, 5, 5, 0, 0, 10, 5, 5, 0, 5, 5, 2, 2, 1,
             "val1", "val2", "val1", 2, 2, 1,
             "sval2", "sval1", 2, 2, 1,
             "5.0 5.0", "val2 val1", "sval2 sval1",
             2, 3]
            ], columns=expected_columns
        )
        # fmt: on

        output_df = table_to_frame(
            self.get_output(self.widget.Outputs.data), include_metas=True
        )
        # remove random since it is not possible to test
        output_df = output_df.loc[:, ~output_df.columns.str.endswith("随机值")]

        pd.testing.assert_frame_equal(
            output_df,
            expected_df,
            check_dtype=False,
            check_column_type=False,
            check_categorical=False,
            atol=1e-3,
        )

    def test_metas_results(self):
        """Test if variable that is in meta in input table remains in metas"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self._set_selection(self.widget.gb_attrs_view, [0, 1])

        output = self.get_output(self.widget.Outputs.data)
        self.assertIn(self.data.domain["svar"], output.domain.metas)

    def test_context(self):
        d = self.data.domain
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assert_aggregations_equal(
            ["平均值", "平均值", "平均值", "串接(Concatenate)", "串接(Concatenate)"]
        )

        self.select_table_rows(self.widget.agg_table_view, [0, 2])
        self.widget.agg_checkboxes["中位数"].click()
        self.assert_aggregations_equal(
            ["平均值, 中位数", "平均值", "平均值, 中位数", "串接(Concatenate)", "串接(Concatenate)"]
        )

        self._set_selection(self.widget.gb_attrs_view, [1, 2])
        self.assertListEqual([d["a"], d["b"]], self.widget.gb_attrs)
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "中位数"},
                d["b"]: {"平均值"},
                d["cvar"]: {"平均值", "中位数"},
                d["dvar"]: {"串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

        # send new data and previous data to check if context restored correctly
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assert_aggregations_equal(
            ["平均值, 中位数", "平均值", "平均值, 中位数", "串接(Concatenate)", "串接(Concatenate)"]
        )
        self._set_selection(self.widget.gb_attrs_view, [1, 2])
        self.assertListEqual([d["a"], d["b"]], self.widget.gb_attrs)
        self.assertDictEqual(
            {
                d["a"]: {"平均值", "中位数"},
                d["b"]: {"平均值"},
                d["cvar"]: {"平均值", "中位数"},
                d["dvar"]: {"串接(Concatenate)"},
                d["svar"]: {"串接(Concatenate)"},
            },
            self.widget.aggregations,
        )

    @patch(
        "Orange.data.aggregate.OrangeTableGroupBy.aggregate",
        Mock(side_effect=ValueError("Test unexpected err")),
    )
    def test_unexpected_error(self):
        """Test if exception in aggregation shown correctly"""

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()

        self.assertTrue(self.widget.Error.unexpected_error.is_shown())
        self.assertEqual(
            str(self.widget.Error.unexpected_error),
            "Test unexpected err",
        )

    def test_time_variable(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        test10_path = os.path.join(
            cur_dir, "..", "..", "..", "tests", "datasets", "test10.tab"
        )
        data = Table.from_file(test10_path)

        # time variable as a group by variable
        self.send_signal(self.widget.Inputs.data, data)
        self._set_selection(self.widget.gb_attrs_view, [1])
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(output))

        # time variable as a grouped variable
        self.send_signal(self.widget.Inputs.data, data)
        self._set_selection(self.widget.gb_attrs_view, [5])
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(2, len(output))

    def test_only_nan_in_group(self):
        data = Table(
            Domain([ContinuousVariable("A"), ContinuousVariable("B")]),
            np.array([[1, np.nan], [2, 1], [1, np.nan], [2, 1]]),
        )
        self.send_signal(self.widget.Inputs.data, data)

        # select feature A as group-by
        self._set_selection(self.widget.gb_attrs_view, [0])
        # select all aggregations for feature B
        self.select_table_rows(self.widget.agg_table_view, [1])
        for cb in self.widget.agg_checkboxes.values():
            while not cb.isChecked():
                cb.click()

        # unselect all aggregations for attr A
        self.select_table_rows(self.widget.agg_table_view, [0])
        for cb in self.widget.agg_checkboxes.values():
            while cb.isChecked():
                cb.click()

        expected_columns = [
            "B - 平均值",
            "B - 中位数",
            "B - 取模",
            "B - 标准差",
            "B - 方差",
            "B - 和",
            "B - 最小值",
            "B - 最大值",
            "B - 跨度",
            "B - 首值",
            "B - 末值",
            "B - 随机值",
            "B - 非缺失数量",
            "B - 数量",
            "B - 非缺失占比",
            "B - 串接(Concatenate)",
            "A",
        ]
        n = np.nan
        expected_df = pd.DataFrame(
            [
                [n, n, n, n, n, 0, n, n, n, n, n, n, 0, 2, 0, "", 1],
                [1, 1, 1, 0, 0, 2, 1, 1, 0, 1, 1, 1, 2, 2, 1, "1.0 1.0", 2],
            ],
            columns=expected_columns,
        )
        output_df = table_to_frame(
            self.get_output(self.widget.Outputs.data), include_metas=True
        )
        pd.testing.assert_frame_equal(
            output_df,
            expected_df,
            check_dtype=False,
            check_column_type=False,
            check_categorical=False,
        )


if __name__ == "__main__":
    unittest.main()
