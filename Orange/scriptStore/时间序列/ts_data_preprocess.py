# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
- time_name|时间特征名称|['date']
- y_name|目标名称|['sales']
- index_names|索引名称|[['store_nbr', 'family', 'date']]
# 周期详见：https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
- period|周期|['D']
- year|起始年份|['2017']
返回值: 

'''

import pandas as pd
from Orange.data.pandas_compat import table_from_frame,table_to_frame
import matplotlib.pyplot as plt 

index_names = in_params['index_names']
time_name = in_params['time_name']
period = in_params['period']
year = in_params['year']
y_name = in_params['y_name']

df = table_to_frame(in_data)

if isinstance(index_names, str):
    df = df.set_index(index_names).to_period(period)
    out_object = df
else:
    df[time_name] = df[time_name].dt.to_period(period)
    df = df.set_index(index_names).sort_index()
    average_df = df.groupby(time_name).mean().squeeze()
    if year != '':
        average_df = average_df.loc[average_df.index.year >= int(year)]
    out_object = average_df

if in_object or in_objects:  
    if in_object:
        in_objects = ['']
        in_objects[0] = in_object
    for func in in_objects:
        if func.__name__ == 'seasonal_plot':
            X = average_df.to_frame()
            X["week"] = X.index.week
            X["day"] = X.index.dayofweek
            func(X, y=y_name, period='week', freq='day')
        elif func.__name__ == 'plot_periodogram':
            func(average_df)

    plt.show()

