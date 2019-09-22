cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Apple: Don't modify install_name when touching RPATH.
if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
endif()
# target_sources: use absolute path for INTERFACE_SOURCES.
if(POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif()
# SWIG: use standard target name.
if(POLICY CMP0078)
    cmake_policy(SET CMP0078 NEW)
endif()
# SWIG: use SWIG_MODULE_NAME property.
if(POLICY CMP0086)
    cmake_policy(SET CMP0086 NEW)
endif()

project(SemanticFusion LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(CUDA REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package( Caffe REQUIRED )

add_subdirectory(${ElasticFusion_DIR}/src ${ElasticFusion_DIR}/build)

file(GLOB_RECURSE srcs src/**/*.cpp)
file(GLOB_RECURSE cuda src/*.cu)
#file(GLOB_RECURSE swig_files src/*.i)

set(CUDA_ARCH_BIN "30 35 50 52 61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
set(CUDA_HOST_COMPILER gcc-5)

#set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall -std=c++11")
set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -std=c++11")

#cuda_add_library(semantic_fusion_cuda_library SHARED ${cuda} )
#CUDA_COMPILE(semantic_fusion_cuda_library ${cuda})
#message(${cuda})

add_library(${CMAKE_PROJECT_NAME} SHARED
        ${srcs}
        ${cuda}
        )

target_link_libraries(${CMAKE_PROJECT_NAME}
        ${CUDA_LIBRARIES}
        ${GLOG_LIBRARY}
        ${OpenCV_LIBS}
        ${OPENNI2_LIBRARY}
        ${Caffe_LIBRARIES}
        efusion
        caffe
        )

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        ${EIGEN_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIR}
        ${OPENNI2_INCLUDE_DIR}
        ${Caffe_INCLUDE_DIRS}
        ${ElasticFusion_DIR}/src
        )



add_executable(SemanticFusionRunner
        src/main.cpp
        )

target_link_libraries(SemanticFusionRunner
        ${CMAKE_PROJECT_NAME}
        )

option(BUILD_PYTHON_BINDINGS "Enable building of python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

#SWIG_ADD_MODULE(SemanticFusion python ${swig_files})
#SWIG_LINK_LIBRARIES(SemanticFusion ${PYTHON_LIBRARIES})


#set_property(SOURCE SemanticFusionInterface.i PROPERTY CPLUSPLUS ON)
##set_property(SOURCE SemanticFusionInterface.i PROPERTY SWIG_MODULE_NAME pySemanticFusionInterface)
##swig_add_library(pySemanticFusionInterface
##        LANGUAGE python
##        OUTPUT_DIR ${CMAKE_BINARY_DIR}/python/${PROJECT_NAME}/semantic_fusion
##        SOURCES SemanticFusionInterface.i)
##set_property(TARGET pySemanticFusionInterface PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)
##target_include_directories(pySemanticFusionInterface
##        PRIVATE
##        ../include
##        ${PYTHON_INCLUDE_DIRS}
##        )
### note: macOS is APPLE and also UNIX !
##if(APPLE)
##    set_target_properties(pySemanticFusionInterface PROPERTIES
##            SUFFIX ".so"
##            INSTALL_RPATH "@loader_path;@loader_path/../../${PROJECT_NAME}/.libs"
##            )
##    set_property(TARGET pySemanticFusionInterface APPEND PROPERTY
##            LINK_FLAGS "-flat_namespace -undefined suppress"
##            )
##elseif(UNIX)
##    set_target_properties(pySemanticFusionInterface PROPERTIES
##            INSTALL_RPATH "$ORIGIN:$ORIGIN/../../${PROJECT_NAME}/.libs"
##            )
##endif()
##target_link_libraries(pySemanticFusionInterface
##        PRIVATE
##        SemanticFusionInterface
##        ${PYTHON_LIBRARIES}
##        )

#SET_SOURCE_FILES_PROPERTIES(src/python/SemanticFusionInterface.i PROPERTIES CPLUSPLUS ON)
##SET_SOURCE_FILES_PROPERTIES(SemanticFusionInterface.i PROPERTIES SWIG_FLAGS "-includeall")
#swig_add_library(TestPy LANGUAGE python SOURCES src/python/SemanticFusionInterface.i)
#
#target_include_directories(TestPy PUBLIC
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
#        ${EIGEN_INCLUDE_DIRS}
#        ${CUDA_INCLUDE_DIRS}
#        ${GLOG_INCLUDE_DIR}
#        ${OPENNI2_INCLUDE_DIR}
#        ${Caffe_INCLUDE_DIRS}
#        ${ElasticFusion_DIR}/src
##        ${PYTHON_INCLUDE_DIRS}
#        )

#SWIG_LINK_LIBRARIES(TestPy ${PYTHON_LIBRARIES})