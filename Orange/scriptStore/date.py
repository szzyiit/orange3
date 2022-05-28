# - 参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
'''
- time|包含时间（小时，分钟，秒）数据|[('是', True, True), ('否', False, False)]
- drop|删除原始日期特征|[('是', True, False), ('否', False, True)]
- someN|随便|[]
'''

import re
import numpy as np
import pandas as pd
from Orange.data.pandas_compat import table_from_frame,table_to_frame
from Orange.data import Domain, Table




def add_datepart(df, drop=in_params['drop'], time=in_params['time']):
    "Helper function that adds columns relevant to a date."
    fldname = df.columns[0]
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
         df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


df = table_to_frame(in_data)
assert(len(df.columns) == 1), "数据需只包括日期列，可以使用‘选择列’小部件选择要分析的列"

add_datepart(df, drop=in_params['drop'], time=in_params['time'])

out_data = table_from_frame(df)
