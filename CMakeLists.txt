cmake_minimum_required(VERSION 3.28)
project(digit_recog LANGUAGES CXX CUDA)

add_executable(main main.cu mnist.hpp neural.hpp cuda_ops.hpp)
add_executable(test test.cu)

if(MSVC)
    target_compile_options(main PRIVATE /W4 /WX)
else()
    target_compile_options(main PRIVATE -Wall -Wextra)
endif()