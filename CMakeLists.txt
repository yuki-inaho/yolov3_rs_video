cmake_minimum_required(VERSION 2.8)
project(yolo_on_cv)
add_executable(yolo_on_cv src/main.cpp src/YOLOManager.cpp)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-O2 -std=gnu++11")

include_directories(
    include
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(realsense2 realsense2 REQUIRED)
include_directories(${realsense2_CFLAGS})
target_link_libraries(yolo_on_cv ${realsense2_LDFLAGS})

pkg_check_modules(OpenCV opencv REQUIRED)
include_directories(${OpenCV_CFLAGS})
target_link_libraries(yolo_on_cv ${OpenCV_LDFLAGS})

