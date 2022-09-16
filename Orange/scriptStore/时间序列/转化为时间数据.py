# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
将软件不认识的时间变得认识
- fldname|时间特征|['Month']
返回值：
datetime数据
'''

import numpy as np
import pandas as pd
from Orange.data.pandas_compat import table_from_frame,table_to_frame



def add_datepart(df, fldname):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    fld_dtype = fld.dtype

    if not np.issubdtype(fld_dtype, np.datetime64):
         df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
         
    return df
    


df = table_to_frame(in_data, include_metas=True)

add_datepart(df, fldname=in_params['fldname'])

out_data = table_from_frame(df)
