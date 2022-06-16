# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
将软件不认识的时间变得认识
返回值：
datetime数据
'''

import re
import numpy as np
import pandas as pd
from Orange.data.pandas_compat import table_from_frame,table_to_frame



def add_datepart(df):
    "Helper function that adds columns relevant to a date."
    fldname = df.columns[0]
    fld = df[fldname]
    fld_dtype = fld.dtype

    if not np.issubdtype(fld_dtype, np.datetime64):
         df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
         
    return df
    


df = table_to_frame(in_data, include_metas=True)
assert(len(df.columns) == 1), "数据需只包括日期列，可以使用‘选择列’小部件选择要分析的列"

add_datepart(df)

out_data = table_from_frame(df)
