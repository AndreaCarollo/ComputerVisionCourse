cmake_minimum_required(VERSION 2.8)
project(Ass2_v1)
find_package(OpenCV REQUIRED)

set(SOURCE_DIR ${CMAKE_HOME_DIRECTORY})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_executable( Ass2_v1  main.cpp )

add_library(mylib STATIC mylib.cpp)

target_link_libraries( Ass2_v1  ${OpenCV_LIBS} mylib )


#  ${PROJECT_SOURCE_DIR}  mylib.hpp