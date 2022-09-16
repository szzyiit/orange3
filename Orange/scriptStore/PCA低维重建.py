# 需要安装scikit-image 和 moviepy
# pip install scikit-image
# pip install moviepy
# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''

'''
from matplotlib import pyplot as plt
from sklearn import decomposition
import numpy as np

M = in_object

u, s, v = decomposition.randomized_svd(M, 2)
low_rank = u @ np.diag(s) @ v

plt.figure(figsize=(12, 12))
plt.title('Original video to images')
plt.imshow(low_rank, cmap='gray')


scale = 13   # Adjust scale to change resolution of image
dims = (int(640 * (scale/100)), int(384 * (scale/100)))

plt.figure()
plt.title('PCA re-constructed')
plt.imshow(np.reshape(low_rank[:,140], dims), cmap='gray')

plt.figure()
plt.title('Residual')
plt.imshow(np.reshape(M[:,350] - low_rank[:,350], dims), cmap='gray')

plt.show()