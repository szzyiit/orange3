# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
将“时间（日期）”特征转换为一系列数值或者分类特征
- time|包含时间（小时，分钟，秒）数据|[('是', True, True), ('否', False, False)]
- drop|删除原始日期特征|[('是', True, False), ('否', False, True)]
- fldname|时间特征|['Month']
返回值：
datesYear: 年份
datesMonth: 月份
datesWeek: 一年的第几周
datesDay: 此月的第几天
datesDayofweek: 此周的第几天
datesDayoyear: 此年的第几天
datesIs_month_end: 是否月底
datesIs_month_start: 是否月初
datesIs_quarter_end: 是否季度底
datesIs_quarter_start: 是否季度初
datesIs_year_end: 是否年底
datesIs_year_start: 是否年初
datesHour: 小时
datesMinute: 分钟
datesSecond: 秒
datesElapsed: 距1970年1月1日0:0:0的秒数 
'''

import re
import numpy as np
import pandas as pd
from Orange.data.pandas_compat import table_from_frame,table_to_frame



def add_datepart(df, drop=in_params['drop'], time=in_params['time'], fldname=in_params['fldname']):
    "Helper function that adds columns relevant to a date."
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


df = table_to_frame(in_data, include_metas=True)

add_datepart(df, drop=in_params['drop'], time=in_params['time'])

out_data = table_from_frame(df)
