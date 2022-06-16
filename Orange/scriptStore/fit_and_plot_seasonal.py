# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|['默认值']
'''
- title_seasonal|周期图标题|['Average Sales']
- title_orig|原始图标题|['Product Sales Frequency Components']
- title_detrend|去除周期图标题|['Deseasonalized']
返回值: 

'''

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

title_seasonal = in_params['title_seasonal']
title_orig = in_params['title_orig']
title_detrend = in_params['title_detrend']

if in_object:
    in_objects = ['']
    in_objects[0] = in_object
    assert(len(in_objects[0]) == 2), '需要传入数据'

plot_func = None
for obj in in_objects:
    if type(obj) is list:
        X = obj[0]
        y = obj[1]
    elif hasattr(obj, '__call__'):
        plot_func = obj

model = LinearRegression().fit(X, y)
y_pred = pd.Series(
    model.predict(X),
    index=X.index,
    name='Fitted',
)

out_object = y_pred

y_pred = pd.Series(model.predict(X), index=X.index)
ax = y.plot(alpha=0.5, title=title_seasonal, ylabel="items sold")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax.legend()

y_deseason = y - y_pred

if plot_func and plot_func.__name__ == 'plot_periodogram':

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))
    ax1 = plot_func(y, ax=ax1)
    ax1.set_title(title_orig)
    ax2 = plot_func(y_deseason, ax=ax2)
    ax2.set_title(title_detrend)

plt.show()

