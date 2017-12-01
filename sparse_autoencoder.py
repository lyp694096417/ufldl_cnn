import numpy as np 
import time 

class SparseAutoEncoder:
    def __init__(self, visible_size, hidden_size, rho, lamda, beta):
        # 对参数进行初始化
        self.visible_size = visible_size    # 输入结点数目
        self.hidden_size = hidden_size      # 隐藏结点数目
        self.rho = rho                      # 隐藏结点的平均激活值
        self.lamda = lamda                  # 权重衰减系数（正则项系数）
        self.beta = beta                    # 稀疏值惩罚项权重

        # 设置划分参数，用于将W1、W2、b1、b2从参数集合theta中拆分出来
        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = 2 * hidden_size * visible_size
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size

        # 对神经网络的连接权重进行初始化
        r = np.sqrt(6) / np.sqrt(visible_size + hidden_size + 1)
        rand = np.random.RandomState(int(time.time()))

        W1 = np.asarray(rand.uniform(low = -r, high = r, size = (hidden_size, visible_size)))
        W2 = np.asarray(rand.uniform(low = -r, high = r, size = (visible_size, hidden_size)))
        b1 = np.zeros((hidden_size, 1))
        b2 = np.zeros((visible_size, 1))

        self.theta = np.concatenate((W1.flatten(), W2.flatten(),b1.flatten(), b2.flatten()))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _KL_divergence(self, x, y):
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

    def sparseAutoEncoderCost(self, theta, input):
        """
        函数功能：计算稀疏自编码器的代价函数
        说明： 由于初始神经网络模型的代价函数J(W,b) = (1/m)*Σ((1/2)*(h(xi)-yi)**2) + λ*Σ(Wij**2)
              前一项为平方误差，后一项为正则项，λ为正则想调节参数
              在稀疏自编码器中，一方面输入等于输出，另一方面考虑稀疏惩罚项，则代价函数重构为如下形式：
              J(W,b) = (1/m)*Σ((1/2)*(h(xi)-xi)**2) + λ*Σ(Wij**2) + 
                       β*Σ(rho*log(rho/rho_cap_i)+(1-rho)*log((1-rho)/(1-rho_cap_i)))
        """
        # 从theta中拆分出参数W和b
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
        # 计算隐藏层结点和输出层结点的激活值
        hidden_layer = self._sigmoid(np.dot(W1, input) + b1)
        output_layer = self._sigmoid(np.dot(W2, hidden_layer) + b2)
        # 求每个隐藏神经元的平均激活度
        rho_cap = np.sum(hidden_layer, axis = 1) / input.shape[1]
        # 计算代价函数中的平方误差项
        diff = output_layer - input
        sum_of_squares_error = 0.5 * np.sum(np.multiply(diff, diff)) / input.shape[1]
        # 计算代价函数中的正则项
        weight_decay = 0.5 * self.lamda * (np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2, W2)))
        # 计算代价函数中的稀疏惩罚项（用KL散度来衡量隐藏神经元与稀疏目标的逼近程度）
        KL_divergence = self.beta * np.sum(self._KL_divergence(self.rho, rho_cap))
        # 代价函数值为上面三项之和
        cost = sum_of_squares_error + weight_decay + KL_divergence
        # 计算每一层的偏差值，用于之后求参数W和b的梯度
        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        del_out = np.mat(np.multiply(diff, np.multiply(output_layer, 1 - output_layer)))
        del_hid = np.multiply(np.dot(np.transpose(W2), del_out) + np.transpose(np.mat(KL_div_grad)),
                              np.multiply(hidden_layer, 1 - hidden_layer))
        # 计算W1、W2、b1、b2的梯度
        W1_grad = np.dot(del_hid, np.transpose(input))
        W2_grad = np.dot(del_out, np.transpose(hidden_layer))
        b1_grad = np.sum(del_hid, axis = 1)
        b2_grad = np.sum(del_out, axis = 1)
        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]
        # 将参数W、b合并到参数theta中
        TW1 = W1_grad.A.flatten()
        TW2 = W2_grad.A.flatten()
        Tb1 = b1_grad.A.flatten()
        Tb2 = b2_grad.A.flatten()
        theta_grad = np.concatenate((TW1, TW2, Tb1, Tb2))
        return [cost, theta_grad]

    def sparseAutoEncoderLinearCost(self, theta, input):
         # 从theta中拆分出参数W和b
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
        # 计算隐藏层结点和输出层结点的激活值
        hidden_layer = self._sigmoid(np.dot(W1, input) + b1)
        output_layer = np.dot(W2, hidden_layer) + b2
        # 求每个隐藏神经元的平均激活度
        rho_cap = np.sum(hidden_layer, axis = 1) / input.shape[1]
        # 计算代价函数中的平方误差项
        diff = output_layer - input
        sum_of_squares_error = 0.5 * np.sum(np.multiply(diff, diff)) / input.shape[1]
        # 计算代价函数中的正则项
        weight_decay = 0.5 * self.lamda * (np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2, W2)))
        # 计算代价函数中的稀疏惩罚项（用KL散度来衡量隐藏神经元与稀疏目标的逼近程度）
        KL_divergence = self.beta * np.sum(self._KL_divergence(self.rho, rho_cap))
        # 代价函数值为上面三项之和
        cost = sum_of_squares_error + weight_decay + KL_divergence
        # 计算每一层的偏差值，用于之后求参数W和b的梯度
        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        del_out = np.mat(diff)
        del_hid = np.multiply(np.dot(np.transpose(W2), del_out) + np.transpose(np.mat(KL_div_grad)),
                              np.multiply(hidden_layer, 1 - hidden_layer))
        # 计算W1、W2、b1、b2的梯度
        W1_grad = np.dot(del_hid, np.transpose(input))
        W2_grad = np.dot(del_out, np.transpose(hidden_layer))
        b1_grad = np.sum(del_hid, axis = 1)
        b2_grad = np.sum(del_out, axis = 1)
        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]
        # 将参数W、b合并到参数theta中
        TW1 = W1_grad.A.flatten()
        TW2 = W2_grad.A.flatten()
        Tb1 = b1_grad.A.flatten()
        Tb2 = b2_grad.A.flatten()
        theta_grad = np.concatenate((TW1, TW2, Tb1, Tb2))
        return [cost, theta_grad]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
