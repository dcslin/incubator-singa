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
#include "../src/model/operation/pooling.h"

#include "gtest/gtest.h"
#include <mkldnn.hpp>

//using singa::Pooling;
//using singa::Shape;

TEST(OperationPooling, Forward) {
  const size_t batchsize = 2, c = 1, h = 3, w = 3;
  const float x[batchsize * c * h * w] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                          7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  singa::Tensor in(singa::Shape{batchsize, c, h, w});
  in.CopyDataFromHostPtr(x, batchsize * c * h * w);

//  Pooling pool;
//  singa::LayerConf conf;
//  singa::PoolingConf *poolconf = conf.mutable_pooling_conf();
//  poolconf->set_pool(singa::PoolingConf_PoolMethod_MAX);
//  poolconf->set_kernel_h(2);
//  poolconf->set_kernel_w(2);
//  poolconf->set_pad_h(0);
//  poolconf->set_pad_w(0);
//  poolconf->set_stride_h(1);
//  poolconf->set_stride_w(1);
//  pool.Setup(Shape{1, 3, 3}, conf);

  singa::Tensor y({2,1,2,2}, in.device(), in.data_type());



  try {
  using namespace mkldnn;

  mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);

  std::vector<primitive> net;
//  memory::dims x_dims={batchsize,c,h,w};
//  memory::dims y_dims={batchsize,c,h,w};
  memory::dims x_dims={batchsize,c,h,w};
  memory::dims y_dims={batchsize,c,2,2};
  memory::dims s_dims={1,1};
  memory::dims k_dims={2,2};
  memory::dims p_dims={0,0};
  auto x_md= memory::desc({x_dims},memory::data_type::f32, memory::format::nchw);
  auto y_md= memory::desc({y_dims},memory::data_type::f32, memory::format::nchw);

//  auto pool_d=pooling_forward::desc(prop_kind::forward_inference, pooling_max, x_md, y_md, s_dims, k_dims, p_dims, p_dims, padding_kind::zero);
  auto pool_d=pooling_forward::desc(forward_inference, pooling_max, x_md, y_md, s_dims, k_dims, p_dims, p_dims, padding_kind::zero);
  auto pool_pd = pooling_forward::primitive_desc(pool_d, eng);

  auto y_mem = memory(pool_pd.dst_primitive_desc());
  auto x_mem = memory({{{x_dims}, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw}, eng}, in.block()->mutable_data());

  y_mem.set_data_handle(y.block()->mutable_data());

  net.push_back( pooling_forward(pool_pd, x_mem, y_mem));
  stream(stream::kind::eager).submit(net).wait();


  // Parameter "flag" does not influence pooling
  const float *outptr1 = y.data<float>();
  // Input: 3*3; kernel: 2*2; stride: 1*1; no padding.
  EXPECT_EQ(8u, y.Size());
  EXPECT_EQ(5.0f, outptr1[0]);
  EXPECT_EQ(6.0f, outptr1[1]);
  EXPECT_EQ(8.0f, outptr1[2]);
  EXPECT_EQ(9.0f, outptr1[3]);
  EXPECT_EQ(5.0f, outptr1[4]);
  EXPECT_EQ(6.0f, outptr1[5]);
  EXPECT_EQ(8.0f, outptr1[6]);
  EXPECT_EQ(9.0f, outptr1[7]);

  }
  catch (mkldnn::error &e) {
    LOG(FATAL) << "MKLDNN pooling fwd" << "Status: " << e.status << " Message: " << e.message;
  }

}

//
//TEST(Pooling, Backward) {
//  // src_data
//  const size_t batchsize = 2, c = 1, src_h = 3, src_w = 3;
//  const float x[batchsize * c * src_h * src_w] = {
//      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
//      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
//  singa::Tensor in(singa::Shape{batchsize, c, src_h, src_w});
//  in.CopyDataFromHostPtr(x, batchsize * c * src_h * src_w);
//
//  Pooling pool;
//  singa::LayerConf conf;
//  singa::PoolingConf *poolconf = conf.mutable_pooling_conf();
//  poolconf->set_pool(singa::PoolingConf_PoolMethod_MAX);
//  poolconf->set_kernel_h(2);
//  poolconf->set_kernel_w(2);
//  poolconf->set_pad_h(0);
//  poolconf->set_pad_w(0);
//  poolconf->set_stride_h(1);
//  poolconf->set_stride_w(1);
//  pool.Setup(Shape{1, 3, 3}, conf);
//
//  singa::Tensor out1 = pool.Forward(singa::kTrain, in);
//
//  // grad
//  const size_t grad_h = 2, grad_w = 2;
//  const float dy[batchsize * c * grad_h * grad_w] = {0.1f, 0.2f, 0.3f, 0.4f,
//                                                     0.1f, 0.2f, 0.3f, 0.4f};
//  singa::Tensor grad(singa::Shape{batchsize, c, grad_h, grad_w});
//  grad.CopyDataFromHostPtr(dy, batchsize * c * grad_h * grad_w);
//
//  const auto ret = pool.Backward(singa::kTrain, grad);
//  singa::Tensor in_grad = ret.first;
//  const float *dx = in_grad.data<float>();
//  EXPECT_EQ(18u, in_grad.Size());
//  EXPECT_EQ(0.0f, dx[0]);
//  EXPECT_EQ(0.0f, dx[1]);
//  EXPECT_EQ(0.0f, dx[2]);
//  EXPECT_EQ(0.0f, dx[3]);
//  EXPECT_EQ(0.1f, dx[4]);
//  EXPECT_EQ(0.2f, dx[5]);
//  EXPECT_EQ(0.0f, dx[6]);
//  EXPECT_EQ(0.3f, dx[7]);
//  EXPECT_EQ(0.4f, dx[8]);
//  EXPECT_EQ(0.0f, dx[9]);
//  EXPECT_EQ(0.0f, dx[10]);
//  EXPECT_EQ(0.0f, dx[11]);
//  EXPECT_EQ(0.0f, dx[12]);
//  EXPECT_EQ(0.1f, dx[13]);
//  EXPECT_EQ(0.2f, dx[14]);
//  EXPECT_EQ(0.0f, dx[15]);
//  EXPECT_EQ(0.3f, dx[16]);
//  EXPECT_EQ(0.4f, dx[17]);
//}

