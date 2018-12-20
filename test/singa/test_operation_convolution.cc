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
#include "singa/singa_config.h"

#ifdef USE_CBLAS

#include "../src/model/operation/convolution.h"

#include "gtest/gtest.h"


/*
TEST(Operation_Convolution, Forward) {
  const size_t batch_size = 2, c = 1, h = 3, w = 3;
  const float x[batch_size * c * h * w] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                          7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  singa::Tensor in(singa::Shape{batch_size, c, h, w});
  in.CopyDataFromHostPtr(x, batch_size * c * h * w);

  const size_t num_filters = 1;
  const size_t kernel_w = 3;
  const size_t kernel_h = 3;
  const std::vector<size_t> stride = {2, 2};
  const std::vector<size_t> padding = {1, 1};
  const bool bias_flag = true;

  const float we[num_filters * kernel_w * kernel_h] = {1.0f, 1.0f, 0.0f,
                                                       0.0f, 0.0f, -1.0f,
                                                       0.0f, 1.0f, 0.0f};
  singa::Tensor weight(singa::Shape{num_filters, num_filters, 3, 3});
  weight.CopyDataFromHostPtr(we, num_filters * num_filters * kernel_w * kernel_h);

  const float b[num_filters] = {1.0f};
  singa::Tensor bias(singa::Shape{num_filters});
  bias.CopyDataFromHostPtr(b, num_filters);


  singa::ConvHandle conv_handle(in, {kernel_w, kernel_h}, stride, padding, c, num_filters, bias_flag);
  singa::Tensor out1 = singa::CpuConvForward(in, weight, bias, conv_handle);

  const float *out_ptr1 = out1.data<float>();
  // Input: 3*3; kernel: 3*3; stride: 2*2; padding: 1*1.
  EXPECT_EQ(8u, out1.Size());

  EXPECT_EQ(3.0f, out_ptr1[0]);
  EXPECT_EQ(7.0f, out_ptr1[1]);
  EXPECT_EQ(-3.0f, out_ptr1[2]);
  EXPECT_EQ(12.0f, out_ptr1[3]);
  EXPECT_EQ(3.0f, out_ptr1[4]);
  EXPECT_EQ(7.0f, out_ptr1[5]);
  EXPECT_EQ(-3.0f, out_ptr1[6]);
  EXPECT_EQ(12.0f, out_ptr1[7]);
}
*/
TEST(Operation_Convolution, Backward) {
  const size_t batch_size = 2, c = 1, h = 3, w = 3;
  const float x[batch_size * c * h * w] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                           7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f,
                                           4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  singa::Tensor in(singa::Shape{batch_size, c, h, w});
  in.CopyDataFromHostPtr(x, batch_size * c * h * w);

  const size_t num_filters = 1;
  const size_t kernel_w = 3;
  const size_t kernel_h = 3;
  const std::vector<size_t> stride = {2, 2};
  const std::vector<size_t> padding = {1, 1};
  const bool bias_flag = true;

  const float we[num_filters * kernel_w * kernel_h] = {1.0f, 1.0f, 0.0f,
                                                       0.0f, 0.0f, -1.0f,
                                                       0.0f, 1.0f, 0.0f};
  singa::Tensor weight(singa::Shape{num_filters, num_filters, 3, 3});
  weight.CopyDataFromHostPtr(we, num_filters * num_filters * kernel_w * kernel_h);

  const float b[num_filters] = {1.0f};
  singa::Tensor bias(singa::Shape{num_filters});
  bias.CopyDataFromHostPtr(b, num_filters);


  singa::ConvHandle conv_handle(in, {kernel_w, kernel_h}, stride, padding, c, num_filters, bias_flag);
  singa::Tensor out1 = singa::CpuConvForward(in, weight, bias, conv_handle);

  // grad
  const size_t grad_h = 2, grad_w = 2;
  const float dy[batch_size * num_filters * grad_h * grad_w] = {
      0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};
  singa::Tensor grad(singa::Shape{batch_size, num_filters, grad_h, grad_w});
  grad.CopyDataFromHostPtr(dy, batch_size * num_filters * grad_h * grad_w);

  singa::Tensor in_grad = singa::CpuConvBackwardx(grad, weight, in, conv_handle);

  const float *dx = in_grad.data<float>();
  const float *wptr = we;
  EXPECT_EQ(18u, in_grad.Size());
  EXPECT_EQ(dy[0] * wptr[4], dx[0]);
  EXPECT_EQ(dy[0] * wptr[5] + dy[1] * wptr[3], dx[1]);
  EXPECT_EQ(dy[1] * wptr[4], dx[2]);
  EXPECT_EQ(dy[0] * wptr[7] + dy[2] * wptr[1], dx[3]);
  EXPECT_EQ(
      dy[0] * wptr[8] + dy[1] * wptr[6] + dy[2] * wptr[2] + dy[3] * wptr[0],
      dx[4]);
  EXPECT_EQ(dy[1] * wptr[7] + dy[3] * wptr[1], dx[5]);
  EXPECT_EQ(dy[2] * wptr[4], dx[6]);
  EXPECT_EQ(dy[2] * wptr[5] + dy[3] * wptr[3], dx[7]);
  EXPECT_EQ(dy[3] * wptr[4], dx[8]);
  EXPECT_EQ(dy[4] * wptr[4], dx[9]);
  EXPECT_EQ(dy[4] * wptr[5] + dy[1] * wptr[3], dx[10]);
  EXPECT_EQ(dy[5] * wptr[4], dx[11]);
  EXPECT_EQ(dy[4] * wptr[7] + dy[2] * wptr[1], dx[12]);
  EXPECT_EQ(
      dy[4] * wptr[8] + dy[5] * wptr[6] + dy[6] * wptr[2] + dy[7] * wptr[0],
      dx[13]);
  EXPECT_EQ(dy[5] * wptr[7] + dy[7] * wptr[1], dx[14]);
  EXPECT_EQ(dy[6] * wptr[4], dx[15]);
  EXPECT_EQ(dy[6] * wptr[5] + dy[7] * wptr[3], dx[16]);
  EXPECT_EQ(dy[7] * wptr[4], dx[17]);

  singa::Tensor dw = singa::CpuConvBackwardW(grad, in, weight, conv_handle);
  singa::Tensor db = singa::CpuConvBackwardb(grad, bias, conv_handle);
  const float *dbptr = db.data<float>();
  EXPECT_FLOAT_EQ(dy[0] + dy[1] + dy[2] + dy[3] + dy[4] + dy[5] + dy[6] + dy[7],
                  dbptr[0]);

  const float *dwptr = dw.data<float>();
  EXPECT_EQ(9u, dw.Size());
  EXPECT_FLOAT_EQ(dy[3] * x[4] + dy[7] * x[13], dwptr[0]);
  EXPECT_FLOAT_EQ(dy[3] * x[5] + dy[7] * x[14] + dy[2] * x[3] + dy[6] * x[12],
                  dwptr[1]);
  EXPECT_FLOAT_EQ(dy[2] * x[4] + dy[6] * x[13], dwptr[2]);
  EXPECT_FLOAT_EQ(dy[1] * x[1] + dy[5] * x[10] + dy[3] * x[7] + dy[7] * x[16],
                  dwptr[3]);
  EXPECT_FLOAT_EQ(dy[0] * x[0] + dy[4] * x[9] + dy[1] * x[2] + dy[5] * x[11] +
                  dy[2] * x[6] + dy[6] * x[15] + dy[3] * x[8] +
                  dy[7] * x[17],
                  dwptr[4]);
  EXPECT_FLOAT_EQ(dy[0] * x[1] + dy[4] * x[10] + dy[2] * x[7] + dy[6] * x[16],
                  dwptr[5]);
  EXPECT_FLOAT_EQ(dy[1] * x[4] + dy[5] * x[13], dwptr[6]);
  EXPECT_FLOAT_EQ(dy[0] * x[3] + dy[4] * x[12] + dy[1] * x[5] + dy[5] * x[14],
                  dwptr[7]);
  EXPECT_FLOAT_EQ(dy[0] * x[4] + dy[4] * x[13], dwptr[8]);
}

#endif  // USE_CBLAS
