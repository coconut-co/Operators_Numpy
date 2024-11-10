cmake_minimum_required(VERSION 3.11)
project(result LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 寻找opencv库
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(result 3_2conv2D.cpp)
target_include_directories(result PUBLIC ${OpenCV_INCLUDE_DIR})
target_link_libraries(result PUBLIC ${OpenCV_LIBS})
