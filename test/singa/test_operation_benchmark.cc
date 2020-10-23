
/************************************************************
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 *************************************************************/
#include <chrono>
#include <iostream>

#include "../src/core/tensor/tensor_math_cuda.h"
#include "../src/model/operation/convolution.h"
#include "gtest/gtest.h"
#include "singa/core/tensor.h"
#include "singa/singa_config.h"

using namespace singa;
using namespace std;
using namespace std::chrono;

#ifdef USE_CUDNN
TEST(OperationBenchmark, cublasGemmStridedBatchedEx) {

  // CUBLAS_CHECK(cublasSgemmStridedBatched(
  //     handle, transa, transb, ncolB, nrowA, ncolA, &alpha, BPtr, ldb, strideB,
  //     APtr, lda, strideA, &beta, CPtr, ldc, strideC, batchCount));
  // CUBLAS_CHECK(cublasSgemmStridedBatched( handle, transa, transb, ncolB, nrowA, ncolA, &alpha, BPtr, ldb, strideB, APtr, lda, strideA, &beta, CPtr, ldc, strideC, batchCount));
}
TEST(OperationBenchmark, CrossEntropyFwd) {
  auto cuda = std::make_shared<singa::CudaGPU>();
  auto ctx = cuda->context(0);
  int bs = 64;
  int dim = 10;
  vector<DataType> dtypes = {kFloat16, kFloat32};

  Tensor t(Shape{bs}, cuda);
  t.SetValue(0.0f);
  t = t.AsType(kInt);

  for (auto dtype : dtypes) {
    Tensor p(Shape{bs, dim}, cuda, dtype);
    Uniform(0.0f, 1.0f, &p);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
      auto l = CrossEntropyFwd(p, t);
      cudaStreamSynchronize(cuda->context(0)->stream);
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << " dtype " << dtype;
    cout << " - " << time_span.count() << " sec";
    cout << endl;
  }
}

TEST(OperationBenchmark, Mult) {
  auto cuda = std::make_shared<singa::CudaGPU>();
  vector<DataType> dtypes = {kFloat32,kFloat16}; // 1.3929ms vs 234.34us
  vector<unsigned long> second_dims = {64 * 100};

  for (auto second_dim : second_dims) {
    cout << endl;
    for (auto dtype : dtypes) {
      Tensor x(Shape{640, second_dim}, cuda, dtype);
      Tensor w(Shape{second_dim, 2048}, cuda, dtype);
      Gaussian(0.0f, 1.0f, &x);
      Gaussian(0.0f, 1.0f, &w);

      high_resolution_clock::time_point t1 = high_resolution_clock::now();

      for (int i = 0; i < 1000; ++i) {
        auto y = Mult(x, w);
        cudaStreamSynchronize(cuda->context(0)->stream);
      }

      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
      cout << " second dim " << second_dim;
      cout << " dtype " << dtype;
      cout << " - " << time_span.count() << " sec";
      cout << endl;
    }
  }
}

TEST(OperationBenchmark, Conv) {
  auto cuda = std::make_shared<singa::CudaGPU>();
  vector<DataType> dtypes = {kFloat16, kFloat32};
  vector<vector<size_t>> kernels{{3, 3}};
  // vector<string> prefers{"tensor_ops", "fastest"};
  vector<string> prefers{"tensor_ops"};
  vector<unsigned long> in_chans{1024};
  int img_hw = 28;
  int out_hw = img_hw-kernels[0][0]+1;
  size_t out_chan = 1024;
  auto has_bias = false;
  int batch = 64;

  vector<size_t> stride{1, 1};
  vector<size_t> padding{0, 0};
  for (auto kernel : kernels) {
    for (auto in_chan : in_chans) {
      for (auto prefer : prefers) {
        cout << endl;
        for (auto dtype : dtypes) {

          Tensor x(Shape{batch, in_chan, img_hw, img_hw}, cuda, dtype);
          Gaussian(0.0f, 1.0f, &x);
          Tensor w(Shape{out_chan, in_chan, kernel[0], kernel[1]}, cuda, dtype);
          Gaussian(0.0f, 1.0f, &w);
          Tensor b(Shape{out_chan}, cuda, dtype);
          Gaussian(0.0f, 1.0f, &b);
          Tensor dy(Shape{batch, in_chan, out_hw, out_hw}, cuda, dtype);
          Gaussian(0.0f, 1.0f, &dy);

          auto h =
              CudnnConvHandle(x, kernel, stride, padding, in_chan, out_chan,
                              has_bias, 1, 1024 * 1024 * 1024, prefer);

          high_resolution_clock::time_point t1 = high_resolution_clock::now();

          for (int i = 0; i < 30; ++i) {
            auto out1 = GpuConvForward(x, w, b, h);
            auto dx = GpuConvBackwardx(dy, w, x, h);
            auto dw = GpuConvBackwardW(dy, x, w, h); // 71.389ms
            cudaDeviceSynchronize();
          }

          high_resolution_clock::time_point t2 = high_resolution_clock::now();
          duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
          cout << " inchan " << in_chan;
          cout << " outchan " << out_chan;
          cout << " ker sz " << kernel[0];
          cout << " prefer " << prefer;
          cout << " dtype " << dtype;
          cout << " - " << time_span.count() << " sec";
          cout << endl;
        }
      }
    }
  }
}
#endif  // USE_CUDNN