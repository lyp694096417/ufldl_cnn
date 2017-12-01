#encoding:utf-8
import numpy as np 
import scipy.signal

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cnn_convolve(patch_dim, n_features, images, W, b, zca_white, patch_mean):
    """
    函数功能：对图像进行卷积
    输入参数: 
    patch_dim:  卷积核的边长维度
    n_features: 特征个数，即每幅图像的大小
    images:     图像数据(shape[0,1，2]维度为图像数据，包括RGB三个通道，shape[3]为图像个数))
    W,b:        编码器参数，将输入层数据映射到隐藏层(或者叫特征层)
    zca_white:  ZCA白化映射矩阵，将原始图像数据进行特征解相关
    patch_mean: 特征均值
    """
    # 图像信息
    n_images = images.shape[3]          # 图像个数
    image_dim = images.shape[0]         # 每幅图像的像素边长
    image_channels = images.shape[2]    # 像素的通道数(RGB = 3)

    convolved_features = np.zeros(shape = (n_features,n_images,image_dim-patch_dim+1,image_dim-patch_dim+1),
                            dtype = np.float)
    WT = W.dot(zca_white)               # 特征矩阵，用于生成卷积核
    bT = b - WT.dot(patch_mean)         # 偏差，调整数据

    for i in range(n_images):
        for j in range(n_features):
            # 存储卷积结果的对象
            convolved_image = np.zeros(shape = (image_dim - patch_dim + 1, image_dim - patch_dim + 1),
                                        dtype = np.float)
            # 分通道进行卷积
            for channel in range(image_channels):
                # 生成卷积核
                patch_size = patch_dim * patch_dim
                feature = WT[j, patch_size * channel:patch_size * (channel + 1)].reshape(patch_dim,patch_dim)
                # 翻转卷积核
                features = np.flipud(np.fliplr(feature))
                # 待卷积图像数据
                im = images[:,:,channel,i]
                # 执行卷积操作
                convolved_image += scipy.signal.convolve2d(im, feature, mode = 'valid')
            # 处理并存储卷积结果
            convolved_image = sigmoid(convolved_image + bT[j])
            convolved_features[j,i,:,:] = convolved_image
    return convolved_features

def cnn_pool(pool_dim, convolved_features):
    """
    函数功能：对卷积结果进行池化操作
    输入参数:
    pool_dim：池化维度
    convolved: 卷积结果
    """
    n_images = convolved_features.shape[1]
    n_features = convolved_features.shape[0]
    convolved_dim = convolved_features.shape[2]
    # 通过assert语句保证池化维度能整除特征图像的维度
    assert convolved_dim % pool_dim == 0,"Pooling dimension is not an exact multiple of convolved dimension"

    pool_size = int(convolved_dim / pool_dim) # 池化维度
    pooled_features = np.zeros(shape = (n_features, n_images, pool_size, pool_size),dtype = np.float)
    for i in range(pool_size):
        for j in range(pool_size):
            pool = convolved_features[:,:,i*pool_dim:(i+1)*pool_dim,j*pool_dim:(j+1)*pool_dim]
            pooled_features[:,:,i,j] = np.mean(np.mean(pool,2),2)
    return pooled_features
