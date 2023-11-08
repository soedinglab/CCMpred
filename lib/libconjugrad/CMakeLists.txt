cmake_minimum_required(VERSION 3.8)
include(CheckLanguage)

project(libconjugrad)
set(LIBCONJUGRAD_MAJOR_VERSION 0)
set(LIBCONJUGRAD_MINOR_VERSION 0)
set(LIBCONJUGRAD_PATCH_VERSION 4)
set(LIBCONJUGRAD_VERSION ${LIBCONJUGRAD_MAJOR_VERSION}.${LIBCONJUGRAD_MINOR_VERSION}.${LIBCONJUGRAD_PATCH_VERSION})

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
	cmake_policy(SET CMP0022 OLD)
endif(CMAKE_VERSION VERSION_GREATER 2.8.11)

set(CMAKE_C_FLAGS "-std=c99 -O3 -g -lm")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

include_directories("${PROJECT_SOURCE_DIR}/include")

set(SOURCES src/conjugrad.c src/debug.c)

if(NOT DEFINED WITH_SIMD)
	set(WITH_SIMD ansi)
endif(NOT DEFINED WITH_SIMD)

if(NOT DEFINED WITH_CUDA)
	set(WITH_CUDA ON)
endif(NOT DEFINED WITH_CUDA)

if(WITH_SIMD STREQUAL ansi)
	add_definitions(-DCONJUGRAD_SIMD=0)
	list(APPEND SOURCES src/arithmetic_ansi.c)
elseif(WITH_SIMD STREQUAL sse)
	add_definitions(-DCONJUGRAD_SIMD=1)
	list(APPEND SOURCES src/arithmetic_simd.c)
elseif(WITH_SIMD STREQUAL avx)
	add_definitions(-DCONJUGRAD_SIMD=2)
	list(APPEND SOURCES src/arithmetic_simd.c)
endif(WITH_SIMD STREQUAL ansi)

set(BUILD_SHARED_LIBS OFF)

if(WITH_CUDA)
	find_package(CUDA)
	check_language(CUDA)
	enable_language(CUDA)

	#list(APPEND SOURCES src/conjugrad_cuda.c src/help_functions.cu src/conjugrad_kernels.cu)
	list(APPEND SOURCES src/conjugrad_cuda.c src/conjugrad_kernels.cu)

	include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif(WITH_CUDA)

add_library(conjugrad ${SOURCES})

install(TARGETS conjugrad DESTINATION lib)

add_executable(rosenbrock EXCLUDE_FROM_ALL samples/rosenbrock.c)
target_link_libraries(rosenbrock conjugrad)

add_executable(polynomial EXCLUDE_FROM_ALL samples/polynomial.c)
target_link_libraries(polynomial conjugrad)

add_executable(sphere EXCLUDE_FROM_ALL samples/sphere.c)
target_link_libraries(sphere conjugrad)
