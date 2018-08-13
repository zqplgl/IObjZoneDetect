cmake_minimum_required(VERSION 3.1)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#set(CMAKE_CXX_FLAGS "-std=c++11")
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
include_directories(/home/zqp/install_lib/include)
include_directories(./include)
include_directories(/usr/include)

find_library(boostsystem_lib boost_system /usr/lib/x86_64-linux-gnu)
find_library(darknet_lib darknet /home/zqp/install_lib/lib)
find_library(caffe_lib caffe /home/zqp/install_lib/lib)
message( STATUS "boost: " ${boostsystem_lib})
message( STATUS "boost: " ${caffe_lib})
message( STATUS "boost: " ${darknet_lib})

find_package(OpenCV 3.2)
add_library(IObjZoneDetect SHARED ${src})
target_link_libraries(IObjZoneDetect ${darknet_lib} ${OpenCV_LIBS} ${caffe_lib} ${boostsystem_lib})


add_executable(testIObjZoneDetect testIObjZoneDetect.cpp)
target_link_libraries(testIObjZoneDetect IObjZoneDetect)

