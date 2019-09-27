cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(Threads REQUIRED)

find_package(PythonLibs REQUIRED)

# OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")

if(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib Found! Are: " ${PYTHON_LIBRARIES})
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${CMAKE_SOURCE_DIR}/inc)
    include_directories(${PROJECT_SOURCE_DIR})
    include_directories(BEFORE "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/my_pt041/lib/python3.6/site-packages/numpy/core/include/numpy")
    include_directories(BEFORE "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/my_pt041/lib/python3.6/site-packages/numpy/core/include")
    link_directories("/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/my_pt041/lib/python3.6/site-packages/numpy/core/lib")

    

    
    add_library(LEDNet_inteface LEDNet.cc)
    target_link_libraries(LEDNet_inteface ${PYTHON_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} dl util)
    
    add_executable(LEDNet_test ${CMAKE_SOURCE_DIR}/src/main.cc)

    target_link_libraries(LEDNet_test LEDNet_inteface )


else(PYTHONLIBS_FOUND)
    message(FATAL_ERROR "Python Lib NOT FOUND!")
endif(PYTHONLIBS_FOUND)