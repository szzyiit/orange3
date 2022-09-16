# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
# 安装：pip install category_encoders
'''
将某（些）分类特征使用某算法转为数值数据
- cat_encoder|转换方法|[('Hash', 'HashingEncoder', False), ('数量', 'CountEncoder', True)]
- n_components|组分数量(仅适用于Hash)|[2]
- cat_name|待转换分类特征|['']
返回值：
转换后的数据
'''
import ast
import category_encoders
from Orange.data.pandas_compat import table_from_frame,table_to_frame

cat_name = in_params['cat_name']
if cat_name.startswith('[') or cat_name.startswith('\'') or cat_name.startswith('\"'):
    cat_name = ast.literal_eval(cat_name)


def encode(df, cat_name, cat_encoder='CountEncoder', n_components=8):
    Encoder = getattr(category_encoders, cat_encoder)
    
    if cat_encoder == 'HashingEncoder':
        try:
            n_components = int(in_params['n_components'])
        except ValueError:
            raise Exception('组分数量 必须是数字')
        enc = Encoder(cols=cat_name, n_components=n_components).fit(df)
    else:
        enc = Encoder(cols=cat_name).fit(df)

    # transform the dataset
    numeric_dataset = enc.transform(df)
    print(numeric_dataset.head())
    return numeric_dataset, enc

df = table_to_frame(in_data, include_metas=False)
print(in_data[:5])

out_df, encoder = encode(df, cat_name=cat_name, cat_encoder=in_params['cat_encoder'], n_components=in_params['n_components'])

out_data = table_from_frame(out_df)
out_object = encoder