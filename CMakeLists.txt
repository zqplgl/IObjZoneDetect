cmake_minimum_required(VERSION 3.1)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "-std=c++11")
file(GLOB src src/*.cpp)

option(GPU "set gpu mode" OFF)
option(CUDNN "use cudnn" OFF)
if(GPU)
    add_definitions(-DGPU)
    message(STATUS "use cuda")
endif()


if(CUDNN)
    add_definitions(-DCUDNN)
    set(CUDNN_DIR /home/zqp/install_lib/cudnn/cudnn_v6/cuda/include)
    include_directories(${CUDNN_DIR})
    message(STATUS "use cudnn")
endif()

include_directories(/home/zqp/mygithub/IObjZoneDetectYoloV3/include)
include_directories(./include)

find_library(darknet_lib darknet /home/zqp/mygithub/darknet)

find_package(OpenCV)
add_library(IObjZoneDetect SHARED ${src})
target_link_libraries(IObjZoneDetect ${darknet_lib} ${OpenCV_LIBS})

