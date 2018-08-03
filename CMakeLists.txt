cmake_minimum_required(VERSION 3.1)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "-std=c++11")
file(GLOB src src/*.cpp)

option(GPU "set gpu mode" ON)
option(CUDNN "use cudnn" ON)
if(GPU)
    add_definitions(-DGPU)
    include_directories(/usr/local/cuda/include)
    message(STATUS "use cuda")
endif()


if(CUDNN)
    add_definitions(-DCUDNN)
    message(STATUS "use cudnn")
endif()

include_directories(/home/zqp/install_lib/include)
include_directories(./include)

find_library(darknet_lib darknet /home/zqp/install_lib/lib)

find_package(OpenCV)
add_library(IObjZoneDetect SHARED ${src} testIObjZoneDetect.cpp)
target_link_libraries(IObjZoneDetect ${darknet_lib} ${OpenCV_LIBS})


add_executable(testIObjZoneDetect testIObjZoneDetect.cpp)
target_link_libraries(testIObjZoneDetect IObjZoneDetect)

