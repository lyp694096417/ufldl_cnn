# encoding:utf-8
import pickle
import numpy as np 
import scipy.io
import cnn
import sys
import time
import datetime
import os.path as p 
import display
import softmax
import sparse_autoencoder
from sparse_autoencoder import SparseAutoEncoder

"""
在前面的练习中，在小块数据上实现了基于稀疏编码器，并训练获得了参数W和b
但在分辨率较高(像素数较多)的图像上使用全连通的神经网络时，训练的时间开销太大。
因此，基于自然图像的固有特性，对单个像素的操作不具有统计意义，但对部分图像块的操作具有统计意义，
对图像实现部分连通网络，即每个隐含模块仅连接输入数据的部分信息，对隐藏层分块进行参数训练，
图像部分信息的提取通过卷积来实现
"""

PATH = p.dirname(__file__)
PATH = p.join(PATH,'data')
# 图像的信息
image_dim = 64      # 图像维度
image_channels = 3  # 图像通道数

# 卷积核边长维度
patch_dim = 8   

# 自稀疏编码器参数
visible_size = patch_dim * patch_dim * image_channels
output_size = visible_size
hidden_size = 400

# ZCA白化的参数
eplsilon = 0.1

# 池化参数
pool_dim = 19

# 调试参数
debug = True

def loadEncoder(filename):
    filepath = p.join(PATH,filename)
    with open(filepath,'rb') as f:
        opt_theta = pickle.load(f)
        zca_white = pickle.load(f)
        patch_mean = pickle.load(f)
    W = opt_theta[0:hidden_size * visible_size].reshape(hidden_size,visible_size)
    b = opt_theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    if debug == True:
        display.displayColorNetwork(np.dot(W,zca_white).transpose())
    return W,b,zca_white,patch_mean

def loadDate(filename,paramName):
    filepath = p.join(PATH,filename)
    dataSet = scipy.io.loadmat(filepath)
    images = dataSet[paramName[0]]
    labels = dataSet[paramName[1]]
    n_images = dataSet[paramName[2]][0][0]
    return images,labels,n_images

def testConvolution(conv_images,W,b,zca_white,patch_mean):
    """
    函数功能：测试卷积操作是否正常
    """
    # step1：检测卷积过程
    convolved_features = cnn.cnn_convolve(patch_dim, hidden_size, conv_images,
                        W, b, zca_white, patch_mean)
    for i in range(1000):
        # 随机选取图像上的起始位置
        feature_num = np.random.randint(0,hidden_size)
        image_num = np.random.randint(0,8)
        image_row = np.random.randint(0,image_dim - patch_dim + 1)
        image_col = np.random.randint(0,image_dim - patch_dim + 1)
        # 根据其实位置截取一块特征图像
        patch = conv_images[image_row:image_row + patch_dim,
                            image_col:image_col + patch_dim,
                            :,image_num]
        # 重构特征图像结构,消除通道维度
        patch = np.concatenate((patch[:,:,0].flatten(),patch[:,:,1].flatten(),patch[:,:,2].flatten()))
        patch = np.reshape(patch,(patch.size,1))
        # 零均值化
        patch = patch - np.tile(patch_mean, (patch.shape[1], 1)).transpose()
        # ZCA白化
        patch = np.dot(zca_white,patch)
        # 将特征图像映射到特征空间(隐藏层)
        W1 = W
        b1 = b.reshape(hidden_size,1)
        features = sparse_autoencoder.sigmoid(np.dot(W1,patch) + b1)
        # 检测卷积获得的特征值与编码器编码得到的特征值是否在误差允许范围内相等
        if abs(features[feature_num,0] - convolved_features[feature_num,image_num,image_row,image_col]) > 1:
            print('Convolved feature does not match activation from autoencoder')
            print('Feature Number      :', feature_num)
            print('Image Number        :', image_num)
            print('Image Row           :', image_row)
            print('Image Column        :', image_col)
            print('Convolved feature   :', convolved_features[feature_num, image_num, image_row, image_col])
            print('Sparse AE feature   :', features[feature_num, 0])
            sys.exit("Convolved feature does not match activation from autoencoder. Exiting...")
    print("Congratulations! Your convolution code passed the test.")

def testPooling():
    """
    函数功能：检测池化操作是否正常
    """
    # 用于进行池化操作的卷积特征图像的边长维度
    t_patch_dim = 8
    # 池化的维度参数
    t_pool_dim = 4
    # 池化后数据的尺寸维度
    pool_size = int(t_patch_dim / t_pool_dim) 
    # 卷积图像的像素总数
    conv_size = t_patch_dim * t_patch_dim
    # 生成用于测试池化的图像数据
    test_matrix = np.arange(conv_size).reshape(t_patch_dim, t_patch_dim)
    # 进行池化操作，计算获得理论上正确的池化结果，结果存储在expected_matrix中
    expected_matrix = np.zeros(shape = (pool_size,pool_size))
    for i in range(pool_size):
        for j in range(pool_size):
            expected_matrix[i,j] = np.mean(test_matrix[i*t_pool_dim:(i+1)*t_pool_dim,
                                                  j*t_pool_dim:(j+1)*t_pool_dim])
    # 对卷积图像维度进行重构，便于使用自己实现的池化函数
    test_matrix = np.reshape(test_matrix,(1,1,t_patch_dim,t_patch_dim))
    # 利用自己实现的池化函数计算的结果
    pooled_features = cnn.cnn_pool(t_pool_dim,test_matrix)
    # 比较池化函数返回结果与理论值是否相等
    if not(pooled_features == expected_matrix).all():
        print("Pooling incorrect")
        print("Expected matrix")
        print(expected_matrix)
        print("Got")
        print(pooled_features)
        sys.exit("Pooling feature does not match expected matrix. Exiting...")
    print('Congratulations! Your pooling code passed the test.')
        
def trainFeatures():
    """
    函数功能: 对原始训练数据和测试数据进行卷积和池化操作，获得隐藏层上的特征数据
    """
    # 载入线性编码器参数
    encoderFile = 'stl10_features.pickle'
    W,b,zca_white,patch_mean = loadEncoder(encoderFile)
    # 载入训练数据
    trainFile = 'stlTrainSubset.mat'
    trainParams = ['trainImages','trainLabels','numTrainImages']
    train_images,train_labels,n_train_images = loadDate(trainFile,trainParams)
    # 载入测试数据
    testFile = 'stlTestSubset.mat'
    testParams = ['testImages','testLabels','numTestImages']
    test_image,test_labels,n_test_images = loadDate(testFile,testParams)
    # 测试实现的卷积和池化函数是否正确
    if debug == True:
        # 使用前8幅图像测试卷积操作是否正常
        conv_images = train_images[:,:,:,0:8]
        testConvolution(conv_images,W,b,zca_white,patch_mean)
        testPooling()
    
    # 在训练数据上利用卷积和池化对隐藏层的特征分块进行训练
    pooled_features_train = np.zeros(shape = (hidden_size,n_train_images,
                                    int(np.floor((image_dim - patch_dim + 1) / pool_dim)),
                                    int(np.floor((image_dim - patch_dim + 1) / pool_dim))),
                                    dtype = np.float)
    pooled_features_test = np.zeros(shape = (hidden_size,n_test_images,
                                    int(np.floor((image_dim - patch_dim + 1) / pool_dim)),
                                    int(np.floor((image_dim - patch_dim + 1) / pool_dim))),
                                    dtype = np.float)    
    # 特征步长
    step_size = 25
    # 检查特征个数能否被步长整除
    assert hidden_size % step_size == 0,"step_size should divide hidden_size"
    feature_part_num = int(hidden_size / step_size)
    start_time = time.time()
    for conv_part in range(feature_part_num):
        feature_start = conv_part * step_size
        feature_end = (conv_part + 1) * step_size
        print('Step:',conv_part,'\nfeatures',feature_start,'to',feature_end)
        # 选取特征参数，用于后续从图像中卷积提取在这些特征上的图像信息
        Wt = W[feature_start:feature_end,:]
        bt = b[feature_start:feature_end]
        # 在训练数据上卷积并池化
        print('Convolving and pooling train_images')
        convolved_features = cnn.cnn_convolve(patch_dim, step_size, train_images,
                                              Wt, bt, zca_white, patch_mean)
        pooled_features = cnn.cnn_pool(pool_dim, convolved_features)
        pooled_features_train[feature_start:feature_end,:,:,:] = pooled_features
        print('Time elapsed:',str(datetime.timedelta(seconds = time.time() - start_time)))
        # 在测试数据上卷积并池化
        print('Convolving and pooling test_images')
        convolved_features = cnn.cnn_convolve(patch_dim, step_size, test_image,
                                              Wt, bt, zca_white, patch_mean)
        pooled_features = cnn.cnn_pool(pool_dim, convolved_features)
        pooled_features_test[feature_start:feature_end,:,:,:] = pooled_features
        print('Time elapsed:',str(datetime.timedelta(seconds = time.time() - start_time)))
    # 保存池化后的特征数据
    print('Saving pooled features...')
    with open(p.join(PATH,'cnn_pooled_features.pickle'),'wb') as f:
        pickle.dump(pooled_features_train,f)
        pickle.dump(pooled_features_test,f)
        pickle.dump(train_labels,f)
        pickle.dump(test_labels,f)
    print('Saved')
    print('Time elapsed:',str(datetime.timedelta(seconds=time.time() - start_time)))

def trainClassifier(n_classes,lamda,images,labels,options):
    """
    函数功能：训练softmax训练器
    """
    # 对数据的维度进行重构以适应softmax训练器的接口
    n_train_images = images.shape[1]
    softmax_images = np.transpose(images, axes = [0,2,3,1])
    softmax_images = softmax_images.reshape((int(softmax_images.size / n_train_images), n_train_images))
    input_size = int(softmax_images.size / n_train_images)
    softmax_labels = labels.flatten() - 1 # 确保标注空间的值域为[0,n_classes-1]
    # 训练softmax分类器
    softmax_module = softmax.softmax_train(input_size,n_classes,lamda,softmax_images,softmax_labels,options)
    #(softmax_theta,softmax_input_size,softmax_n_classes) = softmax_module
    return softmax_module

def testClassifier(model,images,labels):
    """
    函数功能：测试softmax分类器的性能
    """
    # 对数据的维度进行重构以适应softmax的测试接口
    n_test_images = images.shape[1]
    softmax_images = np.transpose(images, axes = [0,2,3,1])
    softmax_images = softmax_images.reshape((int(softmax_images.size / n_test_images), n_test_images))
    softmax_labels = labels.flatten() - 1
    predictions = softmax.softmax_predict(model, softmax_images)
    accuracy = 100 * np.sum(predictions == softmax_labels, dtype=np.float64) / labels.shape[0]
    print("Accuracy: {0:.2f}%".format(accuracy))

def execute():
    # step1:载入先前卷积和池化后的特征数据
    if not(p.exists(p.join(PATH,'cnn_pooled_features.pickle'))):
        trainFeatures()
    with open(p.join(PATH,'cnn_pooled_features.pickle'),'rb') as f:
        pooled_features_train = pickle.load(f)
        pooled_features_test = pickle.load(f)
        train_labels = pickle.load(f)
        test_labels = pickle.load(f)
    
    # step2:利用特征数据来训练softmax分类器
    # 定义参数
    softmax_lambda = 1e-4
    n_classes = 4
    options = {'maxiter':1000, 'disp':False}
    # 训练获得softmax模型参数
    softmax_model = trainClassifier(n_classes,softmax_lambda,pooled_features_train,train_labels,options)
    
    # step3:利用测试数据来测试分类器的准确率
    testClassifier(softmax_model,pooled_features_test,test_labels)
    
if __name__ == '__main__':
    execute()
    






