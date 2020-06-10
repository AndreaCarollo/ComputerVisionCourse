cmake_minimum_required(VERSION 3.1)
# set c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# name of project
project(A1)
# needed package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
# select executable
add_executable(A1 main.cpp)
# add library
add_library(Hungarian Hungarian.cpp)
# add target libraries
target_link_libraries(A1 ${OpenCV_LIBS} Hungarian )