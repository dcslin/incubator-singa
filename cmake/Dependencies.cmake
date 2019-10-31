#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

SET(SINGA_LINKER_LIBS "")

IF(USE_MODULES)
    #IF(USE_SHARED_LIBS)
    #    include(FindProtobuf)
    #    SET(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib")
    #    link_directories(${CMAKE_BINARY_DIR}/lib)
    #    SET(PROTOBUF_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    #    SET(PROTOBUF_LIBRARY "${CMAKE_BINARY_DIR}/lib/libprotobuf.so")
    #    SET(PROTOBUF_PROTOC_LIBRARY "${CMAKE_BINARY_DIR}/lib/libprotoc.so")
    #    SET(PROTOBUF_PROTOC_EXECUTABLE "${CMAKE_BINARY_DIR}/bin/protoc")
    #    INCLUDE_DIRECTORIES(SYSTEM ${PROTOBUF_INCLUDE_DIR})
    #    LIST(APPEND SINGA_LINKER_LIBS ${PROTOBUF_LIBRARY})
    #    #IF(USE_CBLAS)
    #        SET(CBLAS_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    #        SET(CBLAS_LIBRARIES "${CMAKE_BINARY_DIR}/lib/libopenblas.so")
    #        INCLUDE_DIRECTORIES(SYSTEM ${CBLAS_INCLUDE_DIR})
    #        LIST(APPEND SINGA_LINKER_LIBS ${CBLAS_LIBRARIES})
    #ENDIF()
    #ELSE()
    include(FindProtobuf)
    SET(PROTOBUF_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    SET(PROTOBUF_LIBRARY "${CMAKE_BINARY_DIR}/lib/libprotobuf.a")
    SET(PROTOBUF_PROTOC_LIBRARY "${CMAKE_BINARY_DIR}/lib/libprotobuf.a")
    SET(PROTOBUF_PROTOC_EXECUTABLE "${CMAKE_BINARY_DIR}/bin/protoc")
    INCLUDE_DIRECTORIES( ${PROTOBUF_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${PROTOBUF_LIBRARY})
    #IF(USE_CBLAS)
    SET(CBLAS_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    SET(CBLAS_LIBRARIES "${CMAKE_BINARY_DIR}/lib/libopenblas.a")
    INCLUDE_DIRECTORIES( ${CBLAS_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${CBLAS_LIBRARIES})
    #ENDIF()
    #ENDIF()
ELSE()
    FIND_PACKAGE( Protobuf 3.0 REQUIRED )
    #MESSAGE(STATUS "proto libs " ${PROTOBUF_LIBRARY})
    LIST(APPEND SINGA_LINKER_LIBS ${PROTOBUF_LIBRARY})
    #IF(USE_CBLAS)
    FIND_PACKAGE(CBLAS REQUIRED)
    INCLUDE_DIRECTORIES( ${CBLAS_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${CBLAS_LIBRARIES})
    #MESSAGE(STATUS "Found cblas at ${CBLAS_LIBRARIES}")
    #ENDIF()
ENDIF()

#INCLUDE("cmake/ProtoBuf.cmake")
#INCLUDE("cmake/Protobuf.cmake")

FIND_PACKAGE(Glog)
IF(GLOG_FOUND)
    #MESSAGE(STATUS "GLOG FOUND at ${GLOG_INCLUDE_DIR}")
    #ADD_DEFINITIONS("-DUSE_GLOG")
    SET(USE_GLOG TRUE)
    LIST(APPEND SINGA_LINKER_LIBS ${GLOG_LIBRARIES})
    INCLUDE_DIRECTORIES(${GLOG_INCLUDE_DIR})
ENDIF()

IF(USE_LMDB)
    FIND_PACKAGE(LMDB REQUIRED)
    INCLUDE_DIRECTORIES( ${LMDB_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${LMDB_LIBRARIES})
    #MESSAGE(STATUS "FOUND lmdb at ${LMDB_INCLUDE_DIR}")
ENDIF()

IF(USE_CUDA)
    INCLUDE("cmake/Cuda.cmake")
    SET(CNMEM_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    SET(CNMEM_LIBRARY "${CMAKE_BINARY_DIR}/lib/libcnmem.a")
    LIST(APPEND SINGA_LINKER_LIBS ${CNMEM_LIBRARY})
ELSE()
    SET(USE_CUDNN FALSE)
ENDIF()

IF(USE_OPENCL)
    FIND_PACKAGE(OpenCL REQUIRED)
    IF(NOT OPENCL_FOUND)
        MESSAGE(SEND_ERROR "OpenCL was requested, but not found.")
    ELSE()
        INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIR})
        LIST(APPEND SINGA_LINKER_LIBS ${OPENCL_LIBRARIES})
        FIND_PACKAGE(ViennaCL REQUIRED)
        IF(NOT ViennaCL_FOUND)
            MESSAGE(SEND_ERROR "ViennaCL is required if OpenCL is enabled.")
        ELSE()
            #MESSAGE(STATUS "Found ViennaCL headers at ${ViennaCL_INCLUDE_DIR}")
            INCLUDE_DIRECTORIES( ${ViennaCL_INCLUDE_DIR})
            LIST(APPEND SINGA_LINKER_LIBS ${ViennaCL_LIBRARIES})
        ENDIF()
    ENDIF()
ENDIF()

#FIND_PACKAGE(Glog REQUIRED)
#INCLUDE_DIRECTORIES(SYSTEM ${GLOG_INCLUDE_DIRS})
#LIST(APPEND SINGA_LINKER_LIBS ${GLOG_LIBRARIES})
#MESSAGE(STATUS "Found glog at ${GLOG_INCLUDE_DIRS}")

IF(USE_OPENCV)
    FIND_PACKAGE(OpenCV REQUIRED)
    MESSAGE(STATUS "Found OpenCV_${OpenCV_VERSION} at ${OpenCV_INCLUDE_DIRS}")
    INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})
    LIST(APPEND SINGA_LINKER_LIBS ${OpenCV_LIBRARIES})
ENDIF()

#LIST(APPEND SINGA_LINKER_LIBS "/home/wangwei/local/lib/libopenblas.so")
#MESSAGE(STATUS "link lib : " ${SINGA_LINKER_LIBS})

IF(USE_PYTHON)
    IF(USE_PYTHON3)
        set(Python_ADDITIONAL_VERSIONS 3.6 3.5 3.4)
        FIND_PACKAGE(PythonInterp 3 REQUIRED)
        FIND_PACKAGE(PythonLibs 3 REQUIRED)
	    FIND_PACKAGE(SWIG 3.0.10 REQUIRED)
    ELSE()
        FIND_PACKAGE(PythonInterp 2.7 REQUIRED)
        FIND_PACKAGE(PythonLibs 2.7 REQUIRED)
	    FIND_PACKAGE(SWIG 3.0.8 REQUIRED)
    ENDIF()
ENDIF()

IF(USE_JAVA)
    FIND_PACKAGE(Java REQUIRED)
    FIND_PACKAGE(JNI REQUIRED)
    FIND_PACKAGE(SWIG 3.0 REQUIRED)
ENDIF()


IF(USE_MKLDNN)
    FIND_PATH(MKLDNN_INCLUDE_DIR NAME "mkldnn.hpp" PATHS "$ENV{CMAKE_INCLUDE_PATH}")
    FIND_LIBRARY(MKLDNN_LIBRARIES NAME "mkldnn" PATHS "$ENV{CMAKE_LIBRARY_PATH}")
    MESSAGE(STATUS "Found MKLDNN at ${MKLDNN_INCLUDE_DIR}")
    INCLUDE_DIRECTORIES(${MKLDNN_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${MKLDNN_LIBRARIES})
ENDIF()


IF(USE_TC)
    ### Tensor comprehensions
    INCLUDE_DIRECTORIES(/root/TensorComprehensions)
    INCLUDE_DIRECTORIES(/root/TensorComprehensions/tc/version)
    INCLUDE_DIRECTORIES(/root/TensorComprehensions/build)
    # polyhedral model required
    INCLUDE_DIRECTORIES(/root/TensorComprehensions/isl_interface/include)
    # dlpack
    INCLUDE_DIRECTORIES(/root/TensorComprehensions/third-party/dlpack/include)
    # islpp
    INCLUDE_DIRECTORIES(/root/TensorComprehensions/third-party/islpp/include)
    # gflags
    INCLUDE_DIRECTORIES(/root/TensorComprehensions/build/third-party/googlelibraries/gflags/include)
    # glog
    INCLUDE_DIRECTORIES(/root/TensorComprehensions/build/third-party/googlelibraries/glog)
    # Halide
    INCLUDE_DIRECTORIES(/root/conda/envs/tc_build/include/Halide)
    # llvm
    INCLUDE_DIRECTORIES(/root/conda/envs/tc_build/include)
    # torch ATen header
    INCLUDE_DIRECTORIES(/root/conda/envs/tc_build/lib/python3.6/site-packages/torch/lib/include)

    # find Halide lib
    set(HALIDE_PREFIX "/root/conda/envs/tc_build")
    find_library(HALIDE_LIBRARIES REQUIRED NAMES Halide PATHS ${HALIDE_PREFIX} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
    message(STATUS "Found Halide.so file: ${HALIDE_LIBRARIES}")

    # find tc lib
    link_directories(/root/TensorComprehensions/build/tc/aten)
    link_directories(/root/TensorComprehensions/build/tc/lang)
    link_directories(/root/TensorComprehensions/build/tc/core)
    link_directories(/root/TensorComprehensions/build/tc/autotuner)
    link_directories(/root/TensorComprehensions/build/tc/proto)

    # torch(aten)
    link_directories(/root/conda/envs/tc_build/lib/python3.6/site-packages/torch/lib)

    LIST(APPEND SINGA_LINKER_LIBS ${HALIDE_LIBRARIES} tc_aten tc_lang tc_core_cpu tc_cuda tc_core_cuda_no_sdk tc_core tc_autotuner tc_proto ATen)
    ### Tensor comprehensions
ENDIF()
