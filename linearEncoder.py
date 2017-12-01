import numpy as np 
import os.path as p 
import scipy.io
import scipy.optimize
import pickle
import time
import datetime
import display
from sparse_autoencoder import SparseAutoEncoder

# 数据文件存储路径
PATH = p.dirname(__file__)
PATH = p.join(PATH,'data')

# 图像信息
image_channels = 3  # 图像通道数(RGB = 3)
patch_dim = 8       # 图像边长维度
n_patches = 100000  # 样本图像数量

visible_size = patch_dim * patch_dim * image_channels   # 每幅样本图像的像素点个数，也是输入结点数
output_size = visible_size      # 输出结点数目
hidden_size = 400               # 隐藏结点数(也是特征图像的像素点个数)

rho = 0.035                     # 稀疏因子
lamda = 3e-3                    # 正则项系数
beta = 5                        # 稀疏惩罚项系数

epsilon = 0.1                   # 白化正则参数

def execute():
    # 读取图像数据
    filepath = p.join(PATH,'stlSampledPatches.mat')
    patches = scipy.io.loadmat(filepath)['patches']
    # 显示图像数据
    showPic.displayColorNetwork(patches[:,0:100])
    # 零均值化
    patch_mean = np.mean(patches,axis = 1)
    patches = patches - np.tile(patch_mean,(patches.shape[1],1)).transpose()
    # ZCA白化 (Z = BX)
    sigma = np.dot(patches, patches.transpose()) / patches.shape[1]
    (U,S,V) = np.linalg.svd(sigma)
    # B = UΣ**(-1/2)*U.T
    ZCA_white = np.dot(np.dot(U,np.diag(1 / np.sqrt(S + epsilon))), U.transpose())
    patch_ZCAwhite = np.dot(ZCA_white, patches)
    # 显示ZCA白化后的图像
    showPic.displayColorNetwork(patch_ZCAwhite[:,0:100])
    # 创建稀疏编码器
    encoder = SparseAutoEncoder(visible_size, hidden_size, rho, lamda, beta)
    options = {'maxiter': 400, 'disp': True}
    J = lambda x: encoder.sparseAutoEncoderLinearCost(x, patch_ZCAwhite)
    # 训练稀疏编码器获得特征权重W和偏移值b
    start_time = time.time()
    result = scipy.optimize.minimize(J, encoder.theta,method = 'L-BFGS-B',
                                    jac = True, options = options)
    print("Time elapsed:",str(datetime.timedelta(seconds=time.time() - start_time)))
    opt_theta = result.x
    # 存储数据（1 opt_theta 1 白化后的数据 3 均值数据）
    with open(p.join(PATH,'stl10_features.pickle'),'wb') as f:
        pickle.dump(opt_theta, f)
        pickle.dump(ZCA_white, f)
        pickle.dump(patch_mean,f)
    W = opt_theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    b = opt_theta[2*hidden_size * visible_size:2*hidden_size*visible_size + hidden_size]
    showPic.displayColorNetwork(W.dot(ZCA_white).transpose())
    

if __name__ == '__main__':
    #execute()
    with open(p.join(PATH,'stl10_features.pickle'),'rb') as f:
        opt_theta = pickle.load(f)
        zca_white = pickle.load(f)
        patch_mean = pickle.load(f)

    W = opt_theta[0:hidden_size * visible_size].reshape(hidden_size,visible_size)
    b = opt_theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    display.displayColorNetwork(np.dot(W,zca_white).transpose())
    


