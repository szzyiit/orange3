# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
- y_name|目标名称|['FoodAndBeverage']
- time_name|时间特征名称|['Month']
- order|幂数|[1]
- graph_title|图名|['Trend Forecast']
- draw_start| 画图起始时间|["1992-01"]

使用趋势预测时序数据
返回值: 
趋势
'''

from statsmodels.tsa.deterministic import DeterministicProcess
import pandas as pd
from Orange.data.pandas_compat import table_to_frame,table_from_frame
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

y_name = in_params['y_name']
time_name = in_params['time_name']
graph_title = in_params['graph_title']
draw_start = in_params['draw_start']
order = in_params['order']

df = table_to_frame(in_data)
df = df.set_index(time_name).to_period()

dp = DeterministicProcess(
    index=df.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=int(order),             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
in_sample_X = dp.in_sample()
y = df[y_name]

# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model = LinearRegression(fit_intercept=False)
model.fit(in_sample_X, y)

y_pred = pd.Series(model.predict(in_sample_X), index=in_sample_X.index)

# To make a forecast, we apply our model to "out of sample" features. 
# "Out of sample" refers to times outside of the observation period of
# the training data. Here's how we could make a 30-day forecast:
out_sample_X = dp.out_of_sample(steps=30)

y_fore = pd.Series(model.predict(out_sample_X), index=out_sample_X.index)

y_fore.head()

ax = df[draw_start:].plot(title=graph_title)
ax = y_pred[draw_start:].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()

plt.show()

out_data = table_from_frame(y_pred)