# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
将某（些）分类特征使用某算法转为数值数据
- cat_encoder|转换方法|[('Hash', 'HashingEncoder', False), ('数量', 'CountEncoder', True)]
- n_components|组分数量(仅适用于Hash)|[2]
返回值：
转换后的数据
'''
import category_encoders
import pandas as pd
from Orange.data.pandas_compat import table_from_frame,table_to_frame

def encode(df, cat_encoder='CountEncoder', n_components=8):
    fldnames = df.columns

    Encoder = getattr(category_encoders, cat_encoder)
    
    if cat_encoder == 'HashingEncoder':
        try:
            n_components = int(in_params['n_components'])
        except ValueError:
            raise Exception('组分数量 必须是数字')
        enc = Encoder(cols=fldnames, n_components=n_components).fit(df)
    else:
        enc = Encoder(cols=fldnames).fit(df)

    # transform the dataset
    numeric_dataset = enc.transform(df)
    print(numeric_dataset.head())
    return numeric_dataset

df = table_to_frame(in_data, include_metas=True)
print(in_data[:5])

out_df = encode(df, cat_encoder=in_params['cat_encoder'], n_components=in_params['n_components'])

out_data = table_from_frame(out_df)