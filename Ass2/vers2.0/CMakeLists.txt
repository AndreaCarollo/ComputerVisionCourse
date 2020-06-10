cmake_minimum_required(VERSION 3.1)
# set c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# name of project
project(Ass2_v2)
# needed package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# select executable
add_executable(Ass2_v2 main.cpp)
# add target libraries
target_link_libraries(Ass2_v2 ${OpenCV_LIBS})