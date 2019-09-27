#!/usr/bin/python3
#!-*-coding:UTF8-*-

"""
LEDnet 的接口文件
"""
import torch
import os
import numpy as np

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
# 可以考虑留下一个计算预测时间的窗口
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from train.lednet import Net

# TODO 这些东西应该是可以在配置文件中修改的
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# 这个则是为了避免出现另外一种bug
os.environ["GIO_EXTRA_MODULES"]="/usr/lib/x86_64-linux-gnu/gio/modules/"

class LEDNet_Interface:
    def __init__(self, weight_path, num_class):
        """
        初始化LEDnet
        """
        self._weight_path=weight_path
        self._model=Net(num_class)
        self._model=torch.nn.DataParallel(self._model)
        self._model=self._model.cuda()

        # NOTICE 
        torch.set_num_threads(1)
    
        def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            return model

        self._model=load_my_state_dict(self._model, torch.load(weight_path))
    
        print("Model and weights LOADED successfully")
        self._model.eval()

    def eval_image(self, image):
        """
        对一张输入的图像进行评测
        输入的图像应该是numpy格式的图像，大小为(3,512,1024),dtype=float32
        """

        print("[Py|LEDNet_Interface::eval_image] CP1")

        image_tensor2=torch.from_numpy(image)
        with torch.no_grad():
            pre_origin_result = self._model(Variable(image_tensor2.unsqueeze(0).cuda())) 

        print("[Py|LEDNet_Interface::eval_image] CP2")
        
        # 置信度归一化
        pre_unit=torch.nn.functional.softmax(pre_origin_result[0],dim=0)
        # 确定每个像素最可能的类别
        pre=pre_unit.max(0)

        print("[Py|LEDNet_Interface::eval_image] CP3")
    
        # GPU -> CPU
        # 置信度值
        confidence_image=pre[0].cpu().numpy()
        # 分类标签值
        label_image=pre[1].byte().cpu().numpy()

        print("[Py|LEDNet_Interface::eval_image] CP4")

        return confidence_image, label_image


# 下面的内容是给C++程序的接口，之所以有这个接口是因为目前还没有找到C++构造类的实例时，使用有参数的构造函数的办法
def LEDNet_init(weight_path,num_class):
    global model_interface
    model_interface=LEDNet_Interface(weight_path,num_class)
    return

def LEDNet_eval(image):
    print("[Py|LEDNet_eval] enter")
    return model_interface.eval_image(image)


if __name__ == '__main__':

    LEDNet_init('./models/model_best.pth',20)

    datadir = '/home/guoqing/Datasets/KITTI/sequences/08/image_2/000000.png'

    origin_img = cv2.imread(datadir,cv2.IMREAD_COLOR)

    # 在C++的接口函数中需要进行的事情：
    # 1. 如果是使用OpenCV读取的图像，需要将颜色通道转换成为RGB
    img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    # 2. 完成缩放
    img = cv2.resize(img,(1024,512),cv2.INTER_NEAREST)
    # 3. 完成转置操作 shape 从(512,1024,3)  ==> (3,512,1024)
    img = img.transpose((2, 0, 1))
    # 4. 给出的 numpy 应该是float32数据类型（默认的 float 指的是float64）
    image_numpy = img.astype(np.float32)
    # 5. 完成每个通道图像的归一化操作；在C++中可以使用查找表加速， ToTensor的计算速度太慢了，还占用大量CPU计算资源
    image_numpy = image_numpy / 255.0


    c_img, l_img = LEDNet_eval(image_numpy)

    cv2.imshow("origin img",origin_img)
    cv2.imshow("confindence", c_img)
    cv2.imshow("label", l_img*20)

    cv2.waitKey(0)

    print("OK")

