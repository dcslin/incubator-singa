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

#endif  // USE_CBLAS
