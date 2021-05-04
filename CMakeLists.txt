cmake_minimum_required(VERSION 3.2)
project(TEST)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


AUX_SOURCE_DIRECTORY(src DIR_SRCS)
SET(TEST_MATH ${DIR_SRCS})
set(COMMON_INCLUDE include/common)
set(YAML_INCLUDE include/yaml-cpp/include)
set(YAML_LIB_DIR include/yaml-cpp/libs)
set(OPENCV_INCLUDE /mnt/nfs/chengyong/Workspace/compile/opencv-3.4.10/include)

include_directories(include ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${COMMON_INCLUDE} ${YAML_INCLUDE} ${OPENCV_INCLUDE})
link_directories(${YAML_LIB_DIR})

# add_executable(render main.cpp)
ADD_EXECUTABLE(${PROJECT_NAME} ${TEST_MATH} ${COMMON_INCLUDE}/common.cpp)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} yaml-cpp
    nvinfer
    nvinfer_plugin
    nvparsers
    nvonnxparser
    # cudart
)
