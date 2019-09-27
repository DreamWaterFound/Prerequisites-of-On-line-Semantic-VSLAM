/**
 * @file main.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 测试 LEDNET 是否可以正常工作
 * @version 0.1
 * @date 2019-08-16
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include <iostream>
#include <string>
#include <vector>

// #include "LEDNet_interface.hpp"
#include "LEDNet.h"


#include <opencv2/opencv.hpp>

using namespace std;

// 颜色映射关系
const cv::Vec3b colorMap[]=
{
    cv::Vec3b(128, 64,128),
    cv::Vec3b(244, 35,232),
    cv::Vec3b( 70, 70, 70),
    cv::Vec3b(102,102,156),
    cv::Vec3b(190,153,153),

    cv::Vec3b(153,153,153),
    cv::Vec3b(250,170, 30),
    cv::Vec3b(220,220,  0),
    cv::Vec3b(107,142, 35),
    cv::Vec3b(152,251,152),

    cv::Vec3b( 70,130,180),
    cv::Vec3b(220, 20, 60),
    cv::Vec3b(255,  0,  0),
    cv::Vec3b(  0,  0,142),
    cv::Vec3b(  0,  0, 70),

    cv::Vec3b(  0, 60,100),
    cv::Vec3b(  0, 80,100),
    cv::Vec3b(  0,  0,230),
    cv::Vec3b(119, 11, 32),
    cv::Vec3b(  0,  0,  0)
};


int main(int argc, char* argv[])
{
    cout<<"Test LEDNET c++ interface."<<endl;
    cout<<"Complied at "<<__TIME__<<" "<<__DATE__<<"."<<endl;

    LEDNET::LEDNET lednet("/home/guoqing/SLAM/Others/Prerequisites-of-On-line-Semantic-VSLAM/scripts/LEDNET/LEDnet_interface.py",
                          "/home/guoqing/SLAM/Others/Prerequisites-of-On-line-Semantic-VSLAM/scripts/LEDNET/models/model_best.pth",
                          20);

    // LEDNET::LEDNET lednet("/home/guoqing/Codes/043_cpp_call_py_LEDNet/scripts/LEDnet_interface.py",
    //                     "/home/guoqing/Codes/043_cpp_call_py_LEDNet/scripts/models/model_best.pth",
    //                     20);

    if(!lednet.isInitializedResult())
    {
        cout<<"[C++] Something wrong."<<endl;
        cout<<"[C++] "<<lednet.getErrorDescriptionString()<<endl;
        return 0;
    }

    cout<<"[C++] OK."<<endl;

    cv::Mat input_img=cv::imread("/home/guoqing/Datasets/KITTI/sequences/00/image_2/000000.png");

    cv::cvtColor(input_img,input_img,cv::COLOR_BGR2RGB);

    cv::Mat confidence_img, label_img;

    cout<<"[C++] Ready to eval image ..."<<endl;

    // cv::imshow("o_img",input_img);
    // cv::waitKey(0);

    // class LEDNET::PyThreadStateLock PyThreadLock;


    bool res=lednet.evalImage(input_img,confidence_img, label_img);

    if(res)
    {
        cout<<"[C++] Eval OK."<<endl;
    }
    else
    {
        cout<<"[C++] Eval Failed."<<endl;
        cout<<"[C++] "<<lednet.getErrorDescriptionString()<<endl;
        return 0;
    }


    // // 准备绘制带有颜色的结果图像
    cv::Mat coloredImg(label_img.rows,label_img.cols, CV_8UC3);

    size_t min_label=255,max_label=0;
    for(size_t x=0;x<coloredImg.rows;++x)
    {
        for(size_t y=0;y<coloredImg.cols;++y)
        {
            uint8_t label=label_img.at<uint8_t>(x,y);

            if(label<20)
            {
                coloredImg.at<cv::Vec3b>(x,y)=colorMap[label];
            }
            else
            {
                coloredImg.at<cv::Vec3b>(x,y)=cv::Vec3b(0,0,0);
            }

            min_label=label<min_label? label:min_label;
            max_label=label>max_label? label:max_label;
            

        }

    }


    
    cv::imshow("c_img",confidence_img);
    cv::imshow("l_img",label_img*20);
    cv::imshow("r_img",coloredImg);
    
    cv::waitKey(0);
    



    return 0;
    

}


