cmake_minimum_required(VERSION 3.10)

project(CppProject VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})

# Optional include directories if you have a include folder
# target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
