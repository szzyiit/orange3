# 需要安装scikit-image 和 moviepy
# pip install scikit-image
# pip install moviepy
# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
'''

import numpy as np
from Orange.data.pandas_compat import table_from_frame,table_to_frame

df = table_to_frame(in_data) # 你的数据就叫 df 了

df['Time'] = np.arange(len(df.index))
df.head()

out_data = table_from_frame(df) # 将你的数据输出
