cmake_minimum_required (VERSION 2.8)
project (CCMpred)
set(CCMPRED_MAJOR_VERSION 0)
set(CCMPRED_MINOR_VERSION 3)
set(CCMPRED_PATCH_VERSION 2)

set(CCMPRED_VERSION ${CCMPRED_MAJOR_VERSION}.${CCMPRED_MINOR_VERSION}.${CCMPRED_PATCH_VERSION})

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/libconjugrad/CMakeLists.txt")
	message(FATAL_ERROR "CCMpred is missing the libconjugrad submodule! Please make sure to clone CCMpred according to the instructions in the README at \n https://github.com/soedinglab/CCMpred/blob/master/README.md#downloading")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_lib/")

set(CMAKE_C_FLAGS "-std=c99 -O3 -fno-strict-aliasing -g")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/lib/libconjugrad/include")

# OpenMP is on by default. You can disable compiling with OpenMP by using cmake -DWITH_OMP=OFF
if(NOT DEFINED WITH_OMP)
	set(WITH_OMP ON)
endif(NOT DEFINED WITH_OMP)

# CUDA is on by default. You can disable compiling with cuda by using cmake -DWITH_CUDA=OFF
if(NOT DEFINED WITH_CUDA)
	set(WITH_CUDA ON)
endif(NOT DEFINED WITH_CUDA)

# curses is on by default. You can disable compiling with curses by using cmake -DWITH_CURSES=OFF
if(NOT DEFINED WITH_CURSES)
	set(WITH_CURSES ON)
endif(NOT DEFINED WITH_CURSES)

# generate padded code by default. You can disable padding by using cmake -DWITH_PADDING=OFF
if(NOT DEFINED WITH_PADDING)
	set(WITH_PADDING ON)
endif(NOT DEFINED WITH_PADDING)

# enable msgpack support by default. You can disable it by using cmake -DWITH_MSGPACK=OFF
if(NOT DEFINED WITH_MSGPACK)
	set(WITH_MSGPACK ON)
endif(NOT DEFINED WITH_MSGPACK)

# enable jansson support by default. You can disable it by using cmake -DWITH_JANSSON=OFF
if(NOT DEFINED WITH_JANSSON)
	set(WITH_JANSSON ON)
endif(NOT DEFINED WITH_JANSSON)

# enable uuid support by default. You can disable it by using cmake -DWITH_UUID=OFF
if(NOT DEFINED WITH_UUID)
	set(WITH_UUID ON)
endif(NOT DEFINED WITH_UUID)

if(NOT DEFINED CONJUGRAD_FLOAT)
	set(CONJUGRAD_FLOAT 32)
endif(NOT DEFINED CONJUGRAD_FLOAT)

add_definitions("-D_DEFAULT_SOURCE")
add_definitions("-D_GNU_SOURCE")
add_definitions("-DCONJUGRAD_FLOAT=${CONJUGRAD_FLOAT}")
add_definitions("-D__VERSION=\"${CCMPRED_VERSION}\"")
add_definitions("-DBIG_ALIGN=0")

set(SOURCES src/ccmpred.c src/sequence.c src/util.c src/io.c src/evaluate_cpu.c src/parseopt.c src/reweighting.c src/numdif.c src/meta.c)


# Compile with padding if set
if(WITH_PADDING)
	add_definitions(-DPADDING)
endif(WITH_PADDING)


# Compile with msgpack if available
if(WITH_MSGPACK)
	find_package(MsgPack)
endif(WITH_MSGPACK)
if(MSGPACK_FOUND)
	add_definitions("-DMSGPACK")
endif(MSGPACK_FOUND)

# Compile with jansson if available
if(WITH_JANSSON)
	find_package(Jansson)
endif(WITH_JANSSON)
if(LIBJANSSON_FOUND)
	add_definitions("-DJANSSON")
endif(LIBJANSSON_FOUND)


# Compile with uuid if available
if(WITH_UUID)
	find_package(UUID)
endif(WITH_UUID)
if(LIBUUID_FOUND)
	add_definitions("-DUUID")
endif(LIBUUID_FOUND)

# Compile with OpenMP if available
if(WITH_OMP)
	find_package(OpenMP)
endif(WITH_OMP)
if(OPENMP_FOUND)
	list(APPEND SOURCES src/evaluate_cpu_omp.c)
	add_definitions("-DOPENMP")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

# Compile with CUDA if we can find the SDK
if(WITH_CUDA)
	find_package(CUDA)
endif(WITH_CUDA)
if(CUDA_FOUND)
	add_definitions(-DCUDA)

	if(NOT DEFINED GPU_ARCH)
		set(GPU_ARCH "-arch=sm_20")
	endif(NOT DEFINED GPU_ARCH)

	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo ${GPU_ARCH}")

	set(CUDA_SEPARATE_COMPILATION ON)

	cuda_compile(KERNELS_O src/evaluate_cuda_kernels.cu)
	list(APPEND SOURCES src/evaluate_cuda.c ${KERNELS_O})
	include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

# Compile with curses if available
if(WITH_CURSES)
	find_package(Curses)
endif(WITH_CURSES)
if(CURSES_HAVE_CURSES_H AND WITH_CURSES)
	add_definitions(-DCURSES)
endif(CURSES_HAVE_CURSES_H AND WITH_CURSES)



add_executable(ccmpred ${SOURCES})

if(CURSES_HAVE_CURSES_H)
	target_link_libraries(ccmpred ${CURSES_LIBRARIES})
endif(CURSES_HAVE_CURSES_H)


if(JANSSON_LIBRARIES)
	target_link_libraries(ccmpred ${JANSSON_LIBRARIES})
endif(JANSSON_LIBRARIES)


if(UUID_LIBRARIES)
	target_link_libraries(ccmpred ${UUID_LIBRARIES})
endif(UUID_LIBRARIES)

if(UNIX)
	target_link_libraries(ccmpred m)
endif(UNIX)


# compile libConjuGrad
add_subdirectory(lib/libconjugrad)
target_link_libraries(ccmpred conjugrad)


if(CUDA_FOUND)
	set(CMAKE_SKIP_RPATH on) # no -rpath
	set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "") # no -rdynamic
	set(CMAKE_C_LINK_EXECUTABLE "nvcc ${CUDA_NVCC_FLAGS} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -rdc=true -Xcompiler -fopenmp")
	set_target_properties(conjugrad PROPERTIES LINK_LIBRARIES "") # remove absolute path to libcudart.so
endif(CUDA_FOUND)

install(TARGETS ccmpred DESTINATION bin)

# package creation
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Protein residue-residue contacts from correlated mutations predicted accurately and quickly")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
SET(CPACK_PACKAGE_VENDOR "AG Soeding")
set(CPACK_PACKAGE_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${CCMPRED_MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${CCMPRED_MINOR_VERSION}")
set(CPACK_PACKAGE_VERSION_PATCH "${CCMPRED_PATCH_VERSION}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY  "ccmpred ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "ccmpred-${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
if(WIN32 AND NOT UNIX)
	SET(CPACK_GENERATOR NSIS ZIP)
else(WIN32 AND NOT UNIX)
	SET(CPACK_STRIP_FILES "bin/ccmpred")
	SET(CPACK_SOURCE_STRIP_FILES "")

	SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Stefan Seemayer")
	SET(CPACK_DEBIAN_PACKAGE_SECTION "science")

	SET(CPACK_RPM_PACKAGE_LICENSE "GPLv3+")
	SET(CPACK_RPM_PACKAGE_GROUP "Application/Engineering")

	SET(CPACK_GENERATOR DEB RPM)
	SET(CPACK_SOURCE_GENERATOR TGZ)

	SET(CPACK_SOURCE_IGNORE_FILES "_CPack_Packages" ".git" ".mat" "bin" "CMakeFiles" ".zip" ".tar.gz" ".rpm" ".deb" ".swp" "CMakeCache.txt" "CPackSourceConfig.cmake" "CPackConfig.cmake" "cmake_install.cmake" "install_manifest.txt" ".pyc")
endif(WIN32 AND NOT UNIX)
set(CPACK_PACKAGE_EXECUTABLES "CCMpred" "CCMpred")
include(CPack)

