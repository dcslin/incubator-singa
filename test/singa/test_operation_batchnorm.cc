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


TEST(OperationBatchNorm, Forward) {
  const float x_data[] = {1, 2, 3, 4};
  Tensor in(Shape{2, 2});
  in.CopyDataFromHostPtr(x_data, 2 * 2);

  const float y_data[] = {9, 9, 9, 9};
  Tensor y(Shape{2, 2});
  y.CopyDataFromHostPtr(y_data, 2 * 2);

  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {2, 2};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);

  singa::InitLogging("");

  try {
    /* implementation */

    // some useful info: https://github.com/intel/mkl-dnn/issues/367
    using namespace mkldnn;

    // mem
    mkldnn::memory::dims x_dims = {2, 2};
    mkldnn::memory::dims y_dims = {2, 2};
    mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);

    auto x_md = memory::desc(x_dims, mkldnn::memory::data_type::f32, memory::format::nc);

    auto x_mem = memory({{{x_dims}, memory::data_type::f32, memory::format::nc}, eng}, in.block()->mutable_data());
    const memory y_mem = memory({{{y_dims}, memory::data_type::f32, memory::format::nc}, eng}, y.block()->mutable_data());

    // bn fwd operations
    // bn_fwd_d = x_md
    // bn_fwd_pd = bn_fwd_d
    batch_normalization_flag bn_flags = use_scale_shift;
//    auto bn_prop_kind = prop_kind::forward_training;
    prop_kind bn_prop_kind = forward_inference;
    auto bn_fwd_d = batch_normalization_forward::desc(bn_prop_kind, x_md, 1e-5f, bn_flags);
    auto bn_fwd_pd = batch_normalization_forward::primitive_desc(bn_fwd_d, eng);

    const int n_outputs_expected = mkldnn_primitive_desc_query_s32( bn_fwd_pd.get(), mkldnn_query_num_of_outputs_s32, 0);
    EXPECT_EQ(1u, n_outputs_expected);

    auto w_mem = memory(bn_fwd_pd.weights_primitive_desc());
    float w_data[4] = {1, 1, 2, 2};
    w_mem.set_data_handle(w_data);

//    auto m_mem = memory(bn_fwd_pd.mean_primitive_desc());
//    auto v_mem = memory(bn_fwd_pd.variance_primitive_desc());
//    float m_data[4] = {0, 0, 0, 0};
//    float v_data[4] = {0, 0, 0, 0};
//    m_mem.set_data_handle(m_data);
//    v_mem.set_data_handle(v_data);

//    auto bn = batch_normalization_forward(bn_fwd_pd, x_mem, w_mem, y_mem, m_mem, v_mem);
//    auto bn = batch_normalization_forward(bn_fwd_pd, (const primitive::at)x_mem, (const primitive::at)m_mem, (const primitive::at)v_mem, (const primitive::at)w_mem, y_mem);
    auto bn = batch_normalization_forward(bn_fwd_pd, (const primitive::at)x_mem, (const primitive::at)w_mem, y_mem);

    std::vector<primitive> pipeline;
    pipeline.push_back(bn);
    stream(stream::kind::eager).submit(pipeline).wait();

    float *outptr = (float *) y_mem.get_data_handle();
    const auto &shape = y.shape();
    EXPECT_EQ(2u, shape.size());
    EXPECT_EQ(2u, shape[0]);
    EXPECT_EQ(2u, shape[1]);
    EXPECT_NEAR(1.0f, outptr[0], 1e-4f);
    EXPECT_NEAR(1.0f, outptr[1], 1e-4f);
    EXPECT_NEAR(3.0f, outptr[2], 1e-4f);
    EXPECT_NEAR(3.0f, outptr[3], 1e-4f);

  }
  catch (mkldnn::error &e) {
    LOG(FATAL) << "MKLDNN Batch Norm" << "Status: " << e.status << " Message: " << e.message;
  }

  EXPECT_NEAR(1.0f, 1.000001f, 1e-4f);
  /* implementation */
}
/*

TEST(OperationBatchNorm, Backward) {

  const float x[] = {1, 2, 3, 4};
  Tensor in(Shape{2, 2});
  in.CopyDataFromHostPtr(x, 2 * 2);

  const float dy[] = {4, 3, 2, 1};
  Tensor dy_in(Shape{2, 2});
  dy_in.CopyDataFromHostPtr(dy, 2 * 2);

  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {0, 0};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);


  Tensor out = batchnorm.Forward(kTrain, in);
  auto ret = batchnorm.Backward(kTrain, dy_in);
  Tensor dx = ret.first;
  const auto & shape = dx.shape();
  EXPECT_EQ(2u, shape.size());
  EXPECT_EQ(2u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  const float *dxptr = ret.first.data<float>();
  EXPECT_NEAR(.0f, dxptr[0], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[1], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[2], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[3], 1e-4f);

  Tensor dbnScale = ret.second.at(0);
  const float *dbnScaleptr = dbnScale.data<float>();
  const auto & dbnScaleShape = dbnScale.shape();
  EXPECT_EQ(1u, dbnScaleShape.size());
  EXPECT_EQ(2u, dbnScaleShape[0]);

  EXPECT_NEAR(-2.0f, dbnScaleptr[0], 1e-4f);
  EXPECT_NEAR(-2.0f, dbnScaleptr[1], 1e-4f);

  Tensor dbnBias = ret.second.at(1);
  const float *dbnBiasptr = dbnBias.data<float>();
  const auto & dbnBiasShape = dbnBias.shape();
  EXPECT_EQ(1u, dbnBiasShape.size());
  EXPECT_EQ(2u, dbnBiasShape[0]);

  EXPECT_NEAR(6.0f, dbnBiasptr[0], 1e-4f);
  EXPECT_NEAR(4.0f, dbnBiasptr[1], 1e-4f);
}
*/
