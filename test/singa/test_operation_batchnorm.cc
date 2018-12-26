/*********************************************************
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
************************************************************/

#include "../src/model/operation/batchnorm.h"
#include "gtest/gtest.h"
#include <iostream>
#include <mkldnn.hpp>
#include <mkldnn.h>

using namespace singa;

#ifdef USE_MKLDNN

TEST(OperationBatchNorm, Forward) {
  const float x_data[] = {1, 2, 3, 4};
  Tensor in(Shape{2, 2});
  in.CopyDataFromHostPtr(x_data, 2 * 2);

//  const float y_data[] = {9, 9, 9, 9};
//  Tensor y(Shape{2, 2});
//  y.CopyDataFromHostPtr(y_data, 2 * 2);

  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {2, 2};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);

  Tensor moving_mean(Shape{});
  Tensor moving_var(Shape{});


  BatchNormHandle batch_norm_handle(1u,in);
  Tensor y = CpuBatchNormForwardInference(batch_norm_handle, in, alpha, beta, moving_mean,moving_var);



  const float *outptr = y.data<float>();
  const auto &shape = y.shape();
  EXPECT_EQ(2u, shape.size());
  EXPECT_EQ(2u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  EXPECT_NEAR(1.0f, outptr[0], 1e-4f);
  EXPECT_NEAR(1.0f, outptr[1], 1e-4f);
  EXPECT_NEAR(3.0f, outptr[2], 1e-4f);
  EXPECT_NEAR(3.0f, outptr[3], 1e-4f);
}

TEST(OperationBatchNorm, Backward) {
  const float x_data[] = {1, 2, 3, 4};
  Tensor x(Shape{2, 2});
  x.CopyDataFromHostPtr(x_data, 2 * 2);

//  const float dx_data[] = {9, 9, 9, 9};
//  Tensor dx(Shape{2, 2});
//  dx.CopyDataFromHostPtr(dx_data, 2 * 2);

  const float y_data[] = {9, 9, 9, 9};
  Tensor y(Shape{2, 2});
  y.CopyDataFromHostPtr(y_data, 2 * 2);

  const float dy_data[] = {4, 3, 2, 1};
  Tensor dy(Shape{2, 2});
  dy.CopyDataFromHostPtr(dy_data, 2 * 2);

  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {0, 0};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);

//  Tensor dw(Shape{2, 2});
//  dw.CopyDataFromHostPtr(y_data, 2 * 2);


  singa::BatchNormHandle batch_norm_handle(0.0f,x);
  const float running_mean_[] = {0,0};
  Tensor running_mean(Shape{2});
  Tensor running_var(Shape{2});
  running_mean.CopyDataFromHostPtr(running_mean_, 2);
  running_var.CopyDataFromHostPtr(running_mean_, 2);

  auto ret0 = CpuBatchNormForwardTraining(batch_norm_handle, x, alpha, beta, running_mean, running_var);


  Tensor bnScale, bnBias;
  auto  ret =  CpuBatchNormBackwardx( batch_norm_handle, y, dy, x, bnScale,  bnBias, ret0[1],  ret0[2]);

  const auto &shape = ret[0].shape();
  EXPECT_EQ(2u, shape.size());
  EXPECT_EQ(2u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  const float *dxptr = ret[0].data<float>();
  EXPECT_NEAR(.0f, dxptr[0], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[1], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[2], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[3], 1e-4f);


  const auto &dbnScaleShape = ret[1].shape();
  EXPECT_EQ(2u, dbnScaleShape[1]);
  const auto &dbnBiasShape = ret[2].shape();
  EXPECT_EQ(2u, dbnBiasShape[1]);
  const float *dbnScaleptr = ret[1].data<float>();
  EXPECT_NEAR(-2.0f, dbnScaleptr[0], 1e-4f);
  EXPECT_NEAR(-2.0f, dbnScaleptr[1], 1e-4f);
  const float *dbnBiasptr = ret[2].data<float>();
  EXPECT_NEAR(6.0f, dbnBiasptr[0], 1e-4f);
  EXPECT_NEAR(4.0f, dbnBiasptr[1], 1e-4f);
}

#endif // USE_MKLDNN
