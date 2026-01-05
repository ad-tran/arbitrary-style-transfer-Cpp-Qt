cmake_minimum_required(VERSION 3.16)

project(StyleTransferApp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Qt setup
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

# Automatically find all source files in src and subdirectories
file(GLOB_RECURSE SRC_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

add_executable(style_transfer_app ${SRC_FILES})

target_link_libraries(style_transfer_app PRIVATE 
    Qt6::Widgets 
    ${OpenCV_LIBS}
)

# Allow including headers from subdirectories easily
target_include_directories(style_transfer_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
