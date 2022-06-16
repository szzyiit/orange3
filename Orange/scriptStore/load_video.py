# 需要安装scikit-image 和 moviepy
# pip install scikit-image
# pip install moviepy
# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
'''
- file_path|视频位置|[r"D:\OneDrive - HKUST Connect\jupyterNote\ML\fastAI\numerical-linear-algebra-master\nbs\__temp__.mp4"]
'''

from skimage.transform import resize
from skimage import data
import moviepy.editor as mpe
import numpy as np
import scipy
import matplotlib.pyplot as plt

file_path = in_params['file_path']

def create_data_matrix_from_video(clip, k=5, dims=(50, 50)):
    return np.vstack([resize(rgb2gray(clip.get_frame(i/float(k))).astype(int), 
                      dims).flatten() for i in range(k * int(clip.duration))]).T

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def plt_images(M, A, E, index_array, dims, filename=None):
    f = plt.figure(figsize=(15, 10))
    r = len(index_array)
    pics = r * 3
    for k, i in enumerate(index_array):
        for j, mat in enumerate([M, A, E]):
            sp = f.add_subplot(r, 3, 3*k + j + 1)
            sp.axis('Off')
            pixels = mat[:,i]
            if isinstance(pixels, scipy.sparse.csr_matrix):
                pixels = pixels.todense()
            plt.imshow(np.reshape(pixels, dims), cmap='gray')
    return f

def plots(ims, dims, figsize=(15,20), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims)
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        plt.imshow(np.reshape(ims[i], dims), cmap="gray")

scale = 13   # Adjust scale to change resolution of image
dims = (int(640 * (scale/100)), int(384 * (scale/100)))

video = mpe.VideoFileClip(rf'{file_path}')
M = create_data_matrix_from_video(video, 100, dims)

print(dims, M.shape)

plt.imshow(np.reshape(M[:,140], dims), cmap='gray');

plt.figure(figsize=(12, 12))
plt.imshow(M, cmap='gray')
plt.show()

out_object = M