# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
- y_name|目标名称|['sales']
返回值: 

'''

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

y_name = in_params['y_name']
df = in_object

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=df.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()

out_object = [X, df[y_name]]