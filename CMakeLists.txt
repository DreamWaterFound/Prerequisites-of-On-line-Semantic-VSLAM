cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(LEDNet_interface)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 这里使用的是我自己编译的Python版本

set(PYTHON_LIBRARY "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/my_pt041/lib/libpython3.6m.a")
# set(PYTHON_LIBRARY "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/pt041/lib/libpython3.6m.so")

set(PYTHON_INCLUDE_DIR "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/my_pt041/include/python3.6m")


add_subdirectory(src)


