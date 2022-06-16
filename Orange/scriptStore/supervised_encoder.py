# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
################ 注意 ############################
# 如果使用CatBoost，请在先使用“数据采样器”小部件，执行“固定数据比例”为100%的采样，再进行此变换
# 因为CatBoost需要数据行随机打乱
##################################################
'''
将某（些）分类特征使用某算法转为数值数据
- cat_encoder|转换方法|[('目标编码', 'TargetEncoder', True), ('CatBoost', 'CatBoostEncoder', False), ('M estimate', 'MEstimateEncoder', False), ('留一法', 'LeaveOneOutEncoder', False), ('Summary', 'SummaryEncoder', False)]
- percent|训练集比例|[('40%', 0.4, False),('50%', 0.5, True),('60%', 0.6, False),('70%', 0.7, False),('80%', 0.8, False),('90%', 0.9, False),('100%', 1, False)]
返回值：
转换后的数据
'''
import category_encoders
import pandas as pd
import numpy as np
from Orange.data.pandas_compat import table_from_frame,table_to_frame

def encode(df_X, df_y, cat_encoder='TargetEncoder', percent=0.5):
    cut = round(len(df_y) * percent)

    y_train = df_y[:cut]
    y_test = df_y[cut:]

    X_train = df_X[:cut]
    X_test = df_X[cut:]

    Encoder = getattr(category_encoders, cat_encoder)

    # use target encoding to encode two categorical features
    enc = Encoder(cols=df_X.columns)

    # transform the datasets
    training_numeric_array = enc.fit_transform(X_train, y_train)
    testing_numeric_array = enc.transform(X_test)
    
    numeric_array = np.concatenate((training_numeric_array, testing_numeric_array))
    print(numeric_array)
    df = pd.DataFrame(numeric_array, columns = enc.get_feature_names())
     
    return df

# df_X = table_to_frame(in_data.X_df, include_metas=True)
# df_y = table_to_frame(in_data.Y_df)

out_df = encode(in_data.X_df, in_data.Y_df, cat_encoder=in_params['cat_encoder'], percent=in_params['percent'])

out_data = table_from_frame(out_df)
print(out_df.head())