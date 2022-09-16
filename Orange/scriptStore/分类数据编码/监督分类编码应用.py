# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
应用训练完的模型转换分类数据为数值数据
返回值：
转换后的数据
'''
from Orange.data.pandas_compat import table_from_frame

cat_encoder = in_object

df_X = in_data.X_df

# transform the dataset
df = cat_encoder.transform(df_X)

out_df = df.join(in_data.Y_df)

out_data = table_from_frame(out_df)

print(out_df.head(10))
