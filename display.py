import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.figure as fig  
import os.path as p 
import PIL 

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 图像保存路径
PATH = p.dirname(__file__)

"""
实现画板显示图像的类
"""
class ImgPanel:
    def __init__(self, parent):
        # 初始化画板的父容器和初始页面内容
        self.parent = parent
        self.index = 0
        # 创建画板的控制按钮
        btn = Button(parent, text = 'Next', command = self.change)
        # 设置按钮在画板上的位置
        btn.pack(side = BOTTOM, fill = BOTH)
    
    def displayImg(self, images, figsize = (8,8)):
        """
        函数功能： 创建画板并将图像数据载入到画板容器
        参数说明： 
        images： 图像数据
        figsize: 显示图像的初始大小
        """
        # 图像的数量
        n_images = len(images)
        self.maxIndex = n_images
        # 创建画板容器和图像容器
        self.canvas = []
        Figs = []
        for i in range(n_images):
            Figs.append(fig.Figure(figsize))
            self.canvas.append(FigureCanvasTkAgg(Figs[i],master = self.parent))
        self.canvas[0]._tkcanvas.pack()
        # 载入图像数据
        for i,data in enumerate(images):
            axe = Figs[i].add_subplot(111)
            axe.imshow(data, cmap = cm.gray, interpolation = 'bicubic')

    def change(self):
        """
        函数功能： 'Next'按钮的响应函数
        """
        self.index+=1
        if self.index == self.maxIndex:
            self.index = 0
        self.canvas[self.n - 1]._tkcanvas.pack_forget()
        self.canvas[self.n]._tkcanvas.pack()

def normalize(image):
    image = image - np.mean(image)
    std_dev = 3 * np.std(image)
    image = np.maximum(np.minimum(image, std_dev), -std_dev) / std_dev
    image = (image + 1) * 0.5
    return image

def displayNetwork(A, filename = 'features.png'):
    """
    函数功能： 显示灰度图像的特征向量图像
    参数说明： A.T是特征矩阵，输入数据左乘A.T相当与将数据从输入层映射到隐藏层
             故A的shape[0]是输入层结点数，shape[1]是隐藏层结点数
             即每一列是一个特征向量,对应于一副图像
    """
    # 计算特征向量的数量以及每幅特征向量对应图像的行和列
    (n_pixels, n_images) = A.shape
    pixel_dim = int(np.ceil(np.sqrt(n_pixels))) # 特征图像的图像维度
    n_row = int(np.ceil(np.sqrt(n_images)))     # 每幅画布上显示图像的行数
    n_col = int(np.ceil(n_images / n_row))      # 每幅画布上显示图像的列数
    buf = 1                                     # 特征图像在画布上的间隔距离
    images = np.ones(shape = (buf + n_row * (pixel_dim + buf), buf + n_col * (pixel_dim + buf)))

    k = 0
    for i in range(n_row):
        for j in range(n_col):
            if k >= n_images:
                break
            x_i = buf + i * (pixel_dim + buf)
            x_j = buf + j * (pixel_dim + buf)
            y_i = x_i + pixel_dim
            y_j = x_j + pixel_dim
            imgData = normalize(A[:,k])
            images[x_i:y_i, x_j:y_j] = imgData.reshape(pixel_dim,pixel_dim)
            k+=1 
    plt.imshow(images, cmap = cm.gray, interpolation='bicubic')
    plt.show()

def displayColorNetwork(A, filename = 'colorfeatures.png'):
    """
    函数功能：显示RGB图像的特征图像
    """
    # 计算特征向量的数量以及每幅特征向量对应图像的行和列
    (n_pixels, n_images) = A.shape
    n_pixels = int(n_pixels / 3)
    pixel_dim = int(np.ceil(np.sqrt(n_pixels))) # 特征图像的图像维度
    n_row = int(np.ceil(np.sqrt(n_images)))     # 每幅画布上显示图像的行数
    n_col = int(np.ceil(n_images / n_row))      # 每幅画布上显示图像的列数
    buf = 1                    
    # 拆分RGB的三个通道数据
    R = A[0:n_pixels,:]
    G = A[n_pixels:2 * n_pixels,:]
    B = A[2 * n_pixels:3 * n_pixels,:]

    images = np.ones(shape = (buf + n_row * (pixel_dim + buf), buf + n_col * (pixel_dim + buf), 3))

    k = 0
    for i in range(n_row):
        for j in range(n_col):
            if k>=n_images:
                break
            x_i = i * (pixel_dim + buf)
            y_i = x_i + pixel_dim
            x_j = j * (pixel_dim + buf)
            y_j = x_j + pixel_dim
            R_data = normalize(R[:,k])
            G_data = normalize(G[:,k])
            B_data = normalize(B[:,k])
            images[x_i:y_i,x_j:y_j,0] = R_data.reshape(pixel_dim,pixel_dim)
            images[x_i:y_i,x_j:y_j,1] = G_data.reshape(pixel_dim,pixel_dim)
            images[x_i:y_i,x_j:y_j,2] = B_data.reshape(pixel_dim,pixel_dim)
            k+=1
    Fig,axes = plt.subplots(1,1)
    axes.imshow(images)
    axes.set_frame_on(False)
    axes.set_axis_off()
    plt.show()

def showImage(images, figsize = (8,8)):
    root = Tk()
    IS = ImgPanel(root)
    IS.showImages(images,figsize)
    root.mainloop()  
    
        