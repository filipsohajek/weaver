cmake_minimum_required(VERSION 3.30)
project(weaver)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-O3 -march=native -ffast-math -Wall -Wextra")
set(CMAKE_CXX_FLAGS "-g -O3 -DEIGEN_FFTW_DEFAULT -ffast-math -march=native -mtune=native -Wall -Wextra -Wno-ignored-attributes")

include_directories(include)
find_package(Catch2 3 REQUIRED)
add_executable(weaver_tests tests/kernels.cpp)
target_link_libraries(weaver_tests PRIVATE Catch2::Catch2WithMain)
