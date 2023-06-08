cmake_minimum_required(VERSION 3.5)
project(OpenCVLearning)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)

add_subdirectory(src)
