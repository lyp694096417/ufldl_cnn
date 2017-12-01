import numpy as np 
import scipy.sparse 
import scipy.optimize


def softmax_cost(theta, n_classes, input_size, lamda, data, labels):
    """
    参数说明：
    theta: 输入到输出间的权重参数
    n_classes: 输出的结点个数
    input_size: 输入的数据的维度
    lamda: 正则项权重参数
    data: 样本空间数据
    labels: 标注空间数据
    """
    m = data.shape[1]       # 数据的个数，每一列代表隐藏层的一组输出
    theta = theta.reshape(n_classes, input_size)
    theta_data = np.dot(theta, data)
    theta_data = theta_data - np.max(theta_data)
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data),axis=0)
    indicator = scipy.sparse.csr_matrix((np.ones(m),(labels, np.array(range(m)))))
    indicator = np.array(indicator.todense())
    # 计算对数损失模型下的代价函数(正则项是L2正则)
    cost = (-1 / m) * np.sum(indicator * np.log(prob_data)) + (lamda/2) * np.sum(theta * theta)
    # 计算代价函数关于theta的导数
    grad = (-1 / m) * (indicator - prob_data).dot(data.transpose()) + lamda * theta

    return cost,grad.flatten()

def softmax_predict(model, data):
    opt_theta,input_size,n_classes = model
    opt_theta = opt_theta.reshape(n_classes,input_size)
    # 计算输出层的激活值
    prod = np.dot(opt_theta, data)
    # 根据softmax函数将激活值映射到(0,1)间
    pred = np.exp(prod) / np.sum(np.exp(prod),axis=0)
    # 根据输出结点值判断结果
    pred = np.argmax(pred,axis=0)
    return pred

def softmax_train(input_size, n_classes, lamda,data,labels,options = {'maxiter':400,'disp':True}):
    # 利用随机函数初始化参数值
    theta = 0.005 * np.random.randn(n_classes * input_size)
    J = lambda x: softmax_cost(x, n_classes, input_size, lamda, data, labels)
    # 利用拟牛顿迭代法来迭代求解最优参数
    result = scipy.optimize.minimize(J, theta, method = 'L-BFGS-B', jac = True, options = options)
    opt_theta = result.x
    return opt_theta,input_size,n_classes
