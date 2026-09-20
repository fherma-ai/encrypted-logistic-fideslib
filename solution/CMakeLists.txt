# The FIDESlib build of the sign answer: the envelope and the packing are
# C++; the circuit is CUDA. Both libraries are found by a bare find_package
# in the fherma/fideslib image, where /opt/fherma is on CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 3.25.2)
project(solution CXX CUDA)

find_package(OpenFHE REQUIRED)            # the envelope's headers
find_package(fideslib REQUIRED CONFIG)    # brings its own static OpenFHE with it

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
# FIDESlib's flags carry OpenMP; nvcc takes them for the host compiler.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${FIDESLIB_CXX_FLAGS}")

# The four OpenFHE include roots: openfhe.h reaches into binfhe as well.
include_directories(${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include
                    ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke
                    ${OpenFHE_INCLUDE}/binfhe)

add_executable(solution main.cpp solve.cpp gpu.cu)
# The fleet's two cards: the L40S (sm_89) and the RTX PRO 6000 Blackwell
# (sm_120). FIDESlib in the image is compiled for the same two.
set_target_properties(solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89-real;120-real")
# fideslib alone: it links the static OpenFHE into itself, and the shared one
# beside it would be a second copy of every global.
target_link_libraries(solution PRIVATE fideslib::fideslib)
