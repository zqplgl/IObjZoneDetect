cmake_minimum_required(VERSION 3.1)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "-std=c++11")
file(GLOB src src/*.cpp)

option(CPU_ONLY "set cpu mode" OFF)
option(USE_CUDNN "use cudnn" ON)
if(CPU_ONLY)
    add_definitions(-DCPU_ONLY)
    message(STATUS "use cuda")
endif()


if(USE_CUDNN)
    add_definitions(-DUSE_CUDNN)
    message(STATUS "use cudnn")
endif()

include_directories(/usr/local/cuda/include)
include_directories(/usr/include/python3.5m)
#include_directories(/home/zqp/install_lib/include)
include_directories(./include)
include_directories(/usr/include)

find_library(boostsystem_lib boost_system)
find_library(darknet_lib darknet)
find_library(caffe_lib caffe)
find_library(boostpython_lib boost_python3)
find_library(python_lib python3.5m)
message( STATUS "boost: " ${boostsystem_lib})
message( STATUS "boost: " ${boostpython_lib})
message( STATUS "boost: " ${python_lib})
message( STATUS "boost: " ${caffe_lib})
message( STATUS "boost: " ${darknet_lib})

find_package(OpenCV 3.4)
add_library(IObjZoneDetect SHARED ${src})
#add_library(IObjZoneDetect SHARED ${PROJECT_SOURCE_DIR}/src/ObjZoneDetectSSD.cpp)
#add_library(IObjZoneDetect SHARED ${PROJECT_SOURCE_DIR}/src/ObjZoneDetectSSD.cpp)
target_link_libraries(IObjZoneDetect ${darknet_lib} ${OpenCV_LIBS} ${caffe_lib} ${boostsystem_lib})
add_library(_IObjZoneDetect SHARED ${PROJECT_SOURCE_DIR}/python/_IObjZoneDetect.cpp)
target_link_libraries(_IObjZoneDetect ${boostpython_lib} ${python_lib} IObjZoneDetect)


add_executable(testIObjZoneDetect testIObjZoneDetect.cpp)
target_link_libraries(testIObjZoneDetect IObjZoneDetect)

