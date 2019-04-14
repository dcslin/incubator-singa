// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.



// Pass in cmake configurations to swig
#define USE_CUDA 1
#define USE_CUDNN 1
#define USE_OPENCL 0
#define USE_PYTHON 1
#define USE_MKLDNN 1
#define USE_JAVA 0
#define USE_DIST 0
#define CUDNN_VERSION 7401

// SINGA version
#define SINGA_MAJOR_VERSION 1
#define SINGA_MINOR_VERSION 2
/* #undef SINGA_PATCH_VERSION */
#define SINGA_VERSION 1200
