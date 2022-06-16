# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
导入pickle文件
- path|文件路径|['']
返回值: 
包含到指定位置距离的df
'''
import pickle
import numpy as np
import pandas as pd
from Orange.data.pandas_compat import table_from_frame,table_to_frame


with open(rf"{in_params['path']}", "rb") as input_file:
    site_dict = pickle.load(input_file)
    
# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
print(sites_dict.head())

out_data = table_from_frame(sites_dict) # 将你的数据输出