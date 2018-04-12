cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(PARALLEL_MESH LANGUAGES CXX CUDA)


set(SOURCES
	src/grainmesh.h
    src/grainmesh.cpp
    src/iotetgen.h
    src/iotetgen.cpp
	src/mystl.h
	src/mystl.cpp
	src/meshoperations.h
	src/meshoperations.cpp
	src/meshcut.h
	src/meshcut.cpp
	src/meshsmooth.h
	src/meshsmooth.cpp
	src/mymesh.h
	src/mymesh.cpp
    src/common.h
    src/common.cpp
	src/cuda_pip.cuh
	src/helper_math.cuh
	src/cuda_meshgen.cuh
	src/kernel.cu)

add_executable(RJNAMM ${SOURCES})
target_compile_features(RJNAMM PUBLIC cxx_std_11)
set_target_properties( RJNAMM
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET RJNAMM 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
