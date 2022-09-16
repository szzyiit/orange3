# - 参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|')]
'''
延迟n行
- lags|延迟数量|[1]
- fldname|待延迟特征|['']

返回值: 
延迟后的列
'''
import ast
from Orange.data.pandas_compat import table_from_frame,table_to_frame

fldname = in_params['fldname']
if fldname.startswith('\'') or fldname.startswith('\"'):
    fldname = ast.literal_eval(fldname)

df = table_to_frame(in_data)
lags = int(in_params['lags'])


for i in range(1, lags + 1):
    df['Lag_'+str(i)] = df[fldname].shift(i)

df.fillna(0.0, inplace=True)

out_data = table_from_frame(df)