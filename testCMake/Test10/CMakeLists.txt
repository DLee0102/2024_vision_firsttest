cmake_minimum_required(VERSION 3.5)

# Set the project name
project (imported_targets)
# find a boost install with the libraries filesystem and system
#使用库文件系统和系统查找boost install
find_package(Boost 1.46.1 REQUIRED COMPONENTS filesystem system)
#这是第三方库，而不是自己生成的静态动态库
# check if boost was found
if(Boost_FOUND)
    message ("boost found")
else()
    message (FATAL_ERROR "Cannot find Boost")
endif()

# Add an executable
add_executable(imported_targets main.cpp)

# link against the boost libraries
target_link_libraries( imported_targets
    PRIVATE
        Boost::filesystem
)

message(${Boost_INCLUDE_DIR})