cmake_minimum_required(VERSION 3.10)
project(noise_example)

set(CMAKE_CXX_STANDARD 17)

set(EXEC_NAME ${PROJECT_NAME})
add_executable(${EXEC_NAME} main.cpp)
target_include_directories(${EXEC_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})