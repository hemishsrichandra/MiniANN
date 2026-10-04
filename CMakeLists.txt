cmake_minimum_required(VERSION 3.10)
project(MiniANN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_library(miniann ${SOURCES})

# Add tests and examples
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/main.cpp")
    add_executable(miniann_demo examples/main.cpp)
    target_link_libraries(miniann_demo miniann)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_integration.cpp")
    add_executable(miniann_tests tests/test_integration.cpp)
    target_link_libraries(miniann_tests miniann)
endif()
