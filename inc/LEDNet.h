/**
 * @file LEDNet.hpp
 * @author guoqing (1337841346@qq.com)
 * @brief LEDNet的接口函数
 * @version 0.1
 * @date 2019-08-16
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __LEDNET_INTERFACE_HPP__
#define __LEDNET_INTERFACE_HPP__

// C++基础
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

// Python 支持
#include <Python.h>


// TODO 看一下这个注释掉之后会怎么样
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif 

#include "numpy/arrayobject.h"


// OpenCV支持
#include <opencv2/opencv.hpp>

#define EVAL_PY_FUNCTION_NAME       "LEDNet_eval"
#define INIT_PY_FUNCTION_NAME       "LEDNet_init"
#define PY_ENV_PKGS_PATH            "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/pt041/lib/python3.6/site-packages"


namespace LEDNET
{

// 用于解决 Python 中的多线程问题的
 
// class PyThreadStateLock
// {
// public:
//     PyThreadStateLock(void)
//     {
//         state = PyGILState_Ensure( );
//     }
 
//     ~PyThreadStateLock(void)
//     {
//          PyGILState_Release( state );
//     }
// private:
//     PyGILState_STATE state;
// };


class LEDNET
{

public:
    // 构造函数
    LEDNET(const std::string&   pyFilePath,
           const std::string&   modelPath,
           const size_t&        categories);
    // 析构函数
    ~LEDNET();

public:

    // 执行一张图像的预测
    bool evalImage(const cv::Mat&   inputImage,
                         cv::Mat&   confidenceImage,
                         cv::Mat&   labelImage);

public:

     // 查看是否初始化成功
    inline bool isInitializedResult(void) const
    {   return mbIsPythonInitializedOK & mbIsLEDNETInitializedOK;   }

    // 查看错误提示
    inline const std::string& getErrorDescriptionString(void) const
    {   return mstrErrDescription;    }

    // 获取网络可以识别的类别数目
    inline size_t getCLassNum(void) const
    {   return mnCategories;    }

private:

    bool ImportNumPySupport(void) const
    {
        // 这是一个宏，其中包含了返回的语句
        import_array();
        return true;
    }

    bool Image2NumpyCHW_FLOAT(const cv::Mat& srcImage, PyObject*& pPyArray);
    bool Image2NumpyCHW_UBYTE(const cv::Mat& srcImage, PyObject*& pPyArray);
    bool Image2NumpyHWC(const cv::Mat& srcImage, PyObject*& pPyArray);

    bool parseFilePathAndName(const std::string& strFilePathAndName);

private:

    // Python 文件和模块相关
    std::string     mstrPyEnvPkgsPath;              // 指定了 anaconda 寻找文件的路径 site-packages
    std::string     mstrPyMoudlePath;
    std::string     mstrPyMoudleName;
    std::string     mstrEvalPyFunctionName;
    std::string     mstrInitPyFunctionName;
    PyObject*       mpPyEvalModule;
    PyObject*       mpPyEvalFunc;

    // 类别个数
    size_t          mnCategories;

    // 用于在图像转换时的数组指针
    unsigned char  *mpb8ImgTmpArray;
    float          *mpf32ImgTmpArray;

    // 用于指示错误的
    bool            mbIsLEDNETInitializedOK;
    bool            mbIsPythonInitializedOK;
    std::string     mstrErrDescription;

    // 一些中间的文件
    PyObject *mpPyArgList;
    PyObject *mpPyRetValue;
};      // class LEDNET


}       // namespace LEDNET




#endif  // __LEDNET_INTERFACE_HPP__