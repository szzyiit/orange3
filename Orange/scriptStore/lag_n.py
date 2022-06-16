# - 参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|')]
'''
- lags|延迟数量|[1]
延迟n行
返回值: 
延迟后的列
'''

import pandas as pd
from Orange.data.pandas_compat import table_from_frame,table_to_frame

df = table_to_frame(in_data)
lags = int(in_params['lags'])
assert(len(df.columns) == 1), "数据需只包括一列，可以使用‘选择列’小部件选择要分析的列"

fldname = df.columns[0]


for i in range(1, lags + 1):
    df['Lag_'+str(i)] = df[fldname].shift(i)

df.fillna(0.0, inplace=True)

out_data = table_from_frame(df)