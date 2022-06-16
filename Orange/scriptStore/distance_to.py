# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
计算到某地到纽约十个居住条件好的地方的距离
此为示例，具体计算要自己在地图上找相关地点和坐标，对应更改之后可以计算自己的距离
- longitude|经度|['longitude']
- latitude|纬度|['latitude']
返回值: 
包含到指定位置距离的df
'''

import numpy as np
import pandas as pd

from datetime import datetime, timedelta, date
from Orange.data.pandas_compat import table_from_frame,table_to_frame

# Forbes magazine has an article, 
# The Top 10 New York City Neighborhoods to Live In(https://www.forbes.com/sites/trulia/2016/10/04/the-top-10-new-york-city-neighborhoods-to-live-in-according-to-the-locals/#17bf6ff41494)
# According to the Locals, from which we can get neighborhood names.
hoods = {
    "hells" : [40.7622, -73.9924],
    "astoria" : [40.7796684, -73.9215888],
    "Evillage" : [40.723163774, -73.984829394],
    "Wvillage" : [40.73578, -74.00357],
    "LowerEast" : [40.715033, -73.9842724],
    "UpperEast" : [40.768163594, -73.959329496],
    "ParkSlope" : [40.672404, -73.977063],
    "Prospect Park" : [40.93704, -74.17431],
    "Crown Heights" : [40.657830702, -73.940162906],
    "financial" : [40.703830518, -74.005666644],
    "brooklynheights" : [40.7022621909, -73.9871760513],
    "gowanus" : [40.673, -73.997]
}

df = table_to_frame(in_data)
latitude = in_params['latitude']
longitude = in_params['longitude']

for hood, loc in hoods.items():
    # compute manhattan distance
    df[hood] = np.abs(df[latitude] - loc[0]) + np.abs(df[longitude] - loc[1])

out_data = table_from_frame(df)