# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
应用训练完的模型转换分类数据为数值数据
返回值：
转换后的数据
'''
from Orange.data.pandas_compat import table_from_frame,table_to_frame

cat_encoder = in_object

df = table_to_frame(in_data, include_metas=True)

# transform the dataset
numeric_dataset = cat_encoder.transform(df)
print(numeric_dataset.head(10))


out_data = table_from_frame(numeric_dataset)