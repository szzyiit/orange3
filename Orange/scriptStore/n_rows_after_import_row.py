# - 参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|')]
'''
如果每行一个日期，计算在某事件后的几天
假设不能提前知道某特殊事件是否发生，也不知道会持续多久
返回值: 
nth_succesive_row: 已经连续特殊或者正常了多长时间
'''

from typing import List
import numpy as np

import pandas as pd
from Orange.data.pandas_compat import table_from_frame,table_to_frame


def after_important_row(df):
    fldname = df.columns[0]
    fld = df[fldname]
    df[fldname] = df[fldname].astype('category')

    df[fldname] = df[fldname].cat.codes
    print(df[fldname].dtypes)


    '''
    This uses the compare-cumsum-groupby pattern to find the contiguous groups, 
    because df[2].diff().ne(0) gives us a True whenever a value isn't the same as the previous, 
    and the cumulative sum of those gives us a new number whenever a new group of 1s starts.

    This will mean that we have the same group_id for binary values crossing different names, 
    of course, but since we're grouping on both df[0] (the names) and group_ids, we're okay.
    ref: https://stackoverflow.com/questions/50430148/pandas-cumulative-sum-of-consecutive-ones
    '''
    group_ids = df[fldname].diff().ne(0).cumsum()
    df[fldname + 'int'] = df[fldname].astype(int) - 0.5
    df["nth_succesive_row"] = abs(df[fldname+'int'].groupby([group_ids]).cumsum() * 2)
    df.drop(fldname + 'int', axis=1, inplace=True)

    return df

df = table_to_frame(in_data)
assert(len(df.columns) == 1), "数据需只包括一列，可以使用‘选择列’小部件选择要分析的列"

out_df = after_important_row(df)

out_data = table_from_frame(out_df)

print(out_df.head(10))

