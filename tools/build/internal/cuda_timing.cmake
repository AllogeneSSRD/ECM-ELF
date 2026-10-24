find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(CMAKE_CUDA_COMPILER_LAUNCHER "${Python3_EXECUTABLE};${CMAKE_SOURCE_DIR}/tools/build/internal/timed_nvcc.py;--root;${CMAKE_BINARY_DIR};--")
if(CMAKE_CUDA_LINK_EXECUTABLE MATCHES "<CMAKE_CUDA_COMPILER>")
    string(REPLACE "<CMAKE_CUDA_COMPILER>"
        "\"${Python3_EXECUTABLE}\" \"${CMAKE_SOURCE_DIR}/tools/build/internal/timed_nvcc.py\" --root \"${CMAKE_BINARY_DIR}\" -- <CMAKE_CUDA_COMPILER>"
        CMAKE_CUDA_LINK_EXECUTABLE "${CMAKE_CUDA_LINK_EXECUTABLE}")
endif()
