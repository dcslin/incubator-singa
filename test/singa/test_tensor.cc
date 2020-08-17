/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"
#include "singa/core/tensor.h"
#include "singa/core/common.h"
#include "../../src/core/tensor/tensor_math.h"
#include "../../src/core/tensor/tensor_math_cuda.h"
using singa::Device;
using singa::Shape;
using singa::Tensor;

TEST(TensorClass, Constructor) {
  singa::Tensor float_t(singa::Shape{2, 3});
  EXPECT_EQ(6u, float_t.Size());
  EXPECT_EQ(sizeof(float) * 6, float_t.MemSize());
  EXPECT_EQ(singa::kFloat32, float_t.data_type());
  auto s = float_t.shape();
  EXPECT_EQ(s[0], 2u);
  EXPECT_EQ(s[1], 3u);

  EXPECT_NE(float_t.device(), nullptr);

  singa::Tensor float16_t(Shape{2, 3}, singa::kFloat16);
  EXPECT_EQ(singa::kFloat16, float16_t.data_type());
  EXPECT_EQ(6u, float16_t.Size());
  EXPECT_EQ(12u, float16_t.block()->size());

  singa::Tensor x(float16_t);
  EXPECT_EQ(float16_t.Size(), x.Size());
  EXPECT_EQ(float16_t.block(), x.block());
  EXPECT_EQ(float16_t.data_type(), x.data_type());
  EXPECT_EQ(float16_t.device(), x.device());

  singa::Tensor y = float16_t;
  EXPECT_EQ(float16_t.Size(), x.Size());
  EXPECT_EQ(float16_t.block(), x.block());
  EXPECT_EQ(float16_t.data_type(), x.data_type());
  EXPECT_EQ(float16_t.device(), x.device());
}

TEST(TensorClass, Reshape) {
  Tensor t;
  t.Resize(Shape{2, 3});
  EXPECT_TRUE((Shape{2, 3} == t.shape()));

  t.Resize(Shape{3, 3, 4});
  EXPECT_TRUE((Shape{3, 3, 4} == t.shape()));

  t.Resize(Shape{12});
  EXPECT_TRUE((Shape{12} == t.shape()));

  Tensor o;
  EXPECT_TRUE(o.shape() != t.shape());
  o.Resize(Shape{3, 3});
  EXPECT_TRUE(o.shape() != t.shape());
}

TEST(TensorClass, GetValueF16Cpu) {
  using namespace half_float::literal;
  auto cpu = std::make_shared<singa::CppCPU>();
  auto cuda = std::make_shared<singa::CudaGPU>();
  Tensor a({2,3}, cpu, singa::kFloat16);
  a.SetValue(0.1_h);
  a.ToDevice(cuda);
  a.ToDevice(cpu);
  const half_float::half* dptr1 = static_cast<const half_float::half*>(a.block()->data());
  for (int i=0;i<a.size();i++){
    std::cout<<dptr1[i];
  }
}

#ifdef USE_CUDA
TEST(TensorClass, GEMMCuda){
  auto cuda = std::make_shared<singa::CudaGPU>();
  auto cpu = std::make_shared<singa::CppCPU>();

  auto dev = cuda;
  // auto dtype = singa::kFloat16;
  // typedef half_float::half DType;
  auto dtype = singa::kFloat32;
  typedef float DType;
  typedef singa::lang::Cuda Lang;

  Tensor a({2,3},dev,dtype);
  Tensor b({3,2},dev,dtype);
  Tensor out({2,2},dev,dtype);
  a.SetValue(0.111111f);
  b.SetValue(0.333333f);
  out.SetValue(1.1f);
// void GEMM(const DType alpha, const Tensor &A, const Tensor &B, const DType beta, Tensor *C, Context *ctx) {
  // alpha*a*b + beta*c
  singa::GEMM<DType, Lang>(static_cast<DType>(1.0f),a,b,static_cast<DType>(0.0f),&out,dev->context(0));

  out.ToDevice(cpu);
  // auto dptr1 = static_cast<const half_float::half*>(out.block()->data());
  auto dptr1 = static_cast<const DType*>(out.block()->data());
  for(int i=0;i<out.size();i++){
    std::cout<<dptr1[i]<<std::endl;
  }
}

TEST(TensorClass, AsTypeCuda){
  auto cuda = std::make_shared<singa::CudaGPU>();
  auto cpu = std::make_shared<singa::CppCPU>();
  auto dev = cuda;
  auto dtype = singa::kFloat16;
  typedef half_float::half DType;
  // auto dtype = singa::kFloat32;
  // typedef float DType;
  typedef singa::lang::Cuda Lang;
  Tensor a({2,3},dev,dtype);
  a.SetValue(2.2222f);
  // printt(a);
  std::cout << "a "<< a;
}
TEST(TensorClass, UniformFP16Cuda){
  using singa::Context;
  using singa::CudaGPU;
  using singa::CppCPU;
  using singa::kFloat32;
  using singa::kFloat16;
  using namespace singa::lang;
  using singa::CrossEntropyFwd;
  using half_float::half;
  using namespace half_float::literal;

  auto cuda = std::make_shared<CudaGPU>();
  auto cpu = std::make_shared<CppCPU>();
  auto dev = cuda;

  auto dtype = kFloat16;
  typedef half DType;
  // auto dtype = singa::kFloat32;
  // typedef float DType;

  typedef Cuda Lang;
  Tensor p({2,3},dev,dtype);

  Uniform(0.0f,1.0f,&p);
  std::cout<<p;
}
TEST(TensorClass, CrossEntropyBwdFP16Cuda){
  using singa::Context;
  using singa::CudaGPU;
  using singa::CppCPU;
  using singa::kFloat32;
  using singa::kFloat16;
  using namespace singa::lang;
  using singa::CrossEntropyFwd;
  using half_float::half;
  using namespace half_float::literal;

  auto cuda = std::make_shared<CudaGPU>();
  auto cpu = std::make_shared<CppCPU>();
  auto dev = cuda;

  auto dtype = singa::kFloat32;

  typedef Cuda Lang;
  Tensor p({2,3},dev,dtype);
  Tensor t({2,3},dev,dtype);

  float pdata[] = {0.1f, 0.5f, 0.4f, 0.2f, 0.6f, 0.2f};
  p.CopyDataFromHostPtr(pdata, p.size());
  float tdata[] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  t.CopyDataFromHostPtr(tdata, t.size());

  std::cout<<"bef p \n"<<p<<"\n t \n"<<t<< "\n";
  SoftmaxCrossEntropyBwd(t,p);
  std::cout<<"aft p \n"<<p<<"\n t \n"<<t<< "\n";

}

TEST(TensorClass, CrossEntropyFwdFP16Cuda){
  using singa::Context;
  using singa::CudaGPU;
  using singa::CppCPU;
  using singa::kFloat32;
  using singa::kFloat16;
  using namespace singa::lang;
  using singa::CrossEntropyFwd;
  using half_float::half;
  using namespace half_float::literal;

  auto cuda = std::make_shared<CudaGPU>();
  auto cpu = std::make_shared<CppCPU>();
  auto dev = cuda;

  // auto dtype = kFloat16;
  auto dtype = singa::kFloat32;

  typedef Cuda Lang;
  Tensor p({2,3},dev,dtype);
  Tensor t({2,3},dev,dtype);

  float pdata[] = {0.1f, 0.5f, 0.4f, 0.2f, 0.6f, 0.2f};
  p.CopyDataFromHostPtr(pdata, p.size());
  float tdata[] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  t.CopyDataFromHostPtr(tdata, t.size());

  auto l = CrossEntropyFwd(p,t);
  std::cout<<"fp32 p \n"<<p<<"\n t \n"<<t<< "\n l \n" << l << "\n";

  p=p.AsType(kFloat16);
  t=t.AsType(kFloat16);
  // int_target == false
  l = CrossEntropyFwd(p, t);
  std::cout<<"fp16 p \n"<<p<<"\n t \n"<<t<< "\n l \n" << l << "\n";
}

TEST(TensorClass, SoftMaxFP16Cuda){
  using singa::Context;
  using singa::CudaGPU;
  using singa::CppCPU;
  using singa::kFloat32;
  using singa::kFloat16;
  using namespace singa::lang;
  using singa::SoftMax;
  using half_float::half;
  using namespace half_float::literal;

  auto cuda = std::make_shared<CudaGPU>();
  auto cpu = std::make_shared<CppCPU>();
  auto dev = cuda;

  auto dtype = kFloat16;
  typedef half DType;
  // auto dtype = singa::kFloat32;
  // typedef float DType;

  typedef Cuda Lang;
  Tensor a({2,3},dev,dtype);
  Tensor b({2,3},dev,dtype);
  Gaussian(0.0f,1.0f,&a);
  b.SetValue(0.1f);
  SoftMax<DType, Lang>(a, &b, dev->context(0));
  std::cout<<"a\n"<<a<<"\nb\n"<<b<<std::endl;

  a=a.AsType(kFloat32);
  b=b.AsType(kFloat32);
  b.SetValue(0.1f);
  SoftMax<float, Lang>(a, &b, dev->context(0));
  std::cout<<"a\n"<<a<<"\nb\n"<<b<<std::endl;
}

TEST(TensorClass, DotCuda){
  using singa::Context;
  using singa::CudaGPU;
  using singa::CppCPU;
  using singa::kFloat32;
  using singa::kFloat16;
  using namespace singa::lang;
  using singa::Dot;
  using half_float::half;
  using namespace half_float::literal;

  auto cuda = std::make_shared<CudaGPU>();
  auto cpu = std::make_shared<CppCPU>();
  auto dev = cuda;

  auto dtype = kFloat16;
  typedef half DType;
  // auto dtype = singa::kFloat32;
  // typedef float DType;

  typedef Cuda Lang;
  Tensor a({2},dev,dtype);
  Tensor b({2},dev,dtype);
  Tensor out({1},dev,dtype);

  a.SetValue(2.22222f);
  b.SetValue(1.0f);
  out.SetValue(0.0f);
  std::cout<< "a:" << a << std::endl;
  std::cout<< "b:" << b << std::endl;
  std::cout<< "out:" << out << std::endl;

  out.device()->Exec(
    [a, b, out](Context *ctx) mutable { Dot<DType, Lang>(a, b, &out, ctx); },
    {a.block(), b.block()}, {out.block()});
  std::cout<< "out:" << out;
}

TEST(TensorClass, GetValueF16Cuda) {
  using namespace half_float::literal;
  auto cpu = std::make_shared<singa::CppCPU>();
  auto cuda = std::make_shared<singa::CudaGPU>();
  Tensor a({2,3}, cuda, singa::kFloat16);
  a.SetValue(0.33333_h);
  a.ToDevice(cpu);
  const half_float::half* dptr1 = static_cast<const half_float::half*>(a.block()->data());
  // const __half* dptr1 = static_cast<const __half*>(a.block()->data());
  for (int i=0;i<a.size();i++){
    std::cout<<dptr1[i];
  }

  // half_float::half b(0.33333_h);
  // __half b(0.3333f);
  // void* dptr2 = static_cast<void*>(&b);
  // std::cout<< dptr2 << std::endl;

  // __half* dptr3 = static_cast<__half*>(dptr2);
  // std::cout<< dptr3 << *dptr3 << std::endl;

  // half_float::half* dptr = static_cast<half_float::half*>(dptr2);
  // std::cout<< dptr << *dptr << std::endl;

  // copy array
  // Tensor c({2,3}, cuda, singa::kFloat16);
// void Tensor::CopyDataFromHostPtr(const DType *src, const size_t num,
//                                  const size_t offset) const {
  // vector<half_float::half> data_src(c.size(), 0.3333_h);
  // c.CopyDataFromHostPtr(data_src.data(), c.size(), 0);
  // c.ToDevice(cpu);
  // const half_float::half* dptrc = static_cast<const half_float::half*>(c.block()->data());
  // for (int i=0;i<a.size();i++){
  //   std::cout<<"c:"<<dptrc[i];
  // }
}

TEST(TensorClass, FloatAsTypeIntCuda) {
  auto cuda = std::make_shared<singa::CudaGPU>();

  Tensor t(Shape{3}, cuda);
  float data[] = {1.0f, 2.0f, 3.0f};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kFloat32, t.data_type());

  t = t.AsType(singa::kInt);

  EXPECT_EQ(singa::kInt, t.data_type());

  t.ToHost();
  const int* dptr2 = static_cast<const int*>(t.block()->data());
  EXPECT_EQ(1, dptr2[0]);
  EXPECT_EQ(2, dptr2[1]);
  EXPECT_EQ(3, dptr2[2]);
}

TEST(TensorClass, IntAsTypeFloatCuda) {
  auto cuda = std::make_shared<singa::CudaGPU>();

  Tensor t(Shape{3}, cuda, singa::kInt);
  int data[] = {1, 2, 3};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kInt, t.data_type());

  t = t.AsType(singa::kFloat32);

  EXPECT_EQ(singa::kFloat32, t.data_type());

  t.ToHost();
  const float* dptr2 = static_cast<const float*>(t.block()->data());
  EXPECT_EQ(1.0f, dptr2[0]);
  EXPECT_EQ(2.0f, dptr2[1]);
  EXPECT_EQ(3.0f, dptr2[2]);
}

#endif  // USE_CUDA

TEST(TensorClass, FloatAsTypeFloatCPU) {
  Tensor t(Shape{3});
  float data[] = {1.0f, 2.0f, 3.0f};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kFloat32, t.data_type());
  const float* dptr = static_cast<const float*>(t.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);

  Tensor t2 = t.AsType(singa::kFloat32);

  EXPECT_EQ(singa::kFloat32, t2.data_type());

  const float* dptr2 = static_cast<const float*>(t2.block()->data());
  EXPECT_EQ(1.0f, dptr2[0]);
  EXPECT_EQ(2.0f, dptr2[1]);
  EXPECT_EQ(3.0f, dptr2[2]);
}

TEST(TensorClass, FloatAsTypeIntCPU) {
  Tensor t(Shape{3});
  float data[] = {1.0f, 2.0f, 3.0f};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kFloat32, t.data_type());
  const float* dptr = static_cast<const float*>(t.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);

  Tensor t2 = t.AsType(singa::kInt);

  EXPECT_EQ(singa::kInt, t2.data_type());
  const int* dptr2 = static_cast<const int*>(t2.block()->data());
  EXPECT_EQ(1, dptr2[0]);
  EXPECT_EQ(2, dptr2[1]);
  EXPECT_EQ(3, dptr2[2]);
}

TEST(TensorClass, IntAsTypeFloatCPU) {
  Tensor t(Shape{3}, singa::kInt);
  int data[] = {1, 2, 3};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kInt, t.data_type());

  auto t2 = t.AsType(singa::kFloat32);

  EXPECT_EQ(singa::kFloat32, t2.data_type());

  const float* dptr2 = static_cast<const float*>(t2.block()->data());
  EXPECT_EQ(1.0f, dptr2[0]);
  EXPECT_EQ(2.0f, dptr2[1]);
  EXPECT_EQ(3.0f, dptr2[2]);
}

TEST(TensorClass, ToDevice) {
  Tensor t(Shape{2, 3});
  EXPECT_EQ(singa::defaultDevice, t.device());
  auto dev = std::make_shared<singa::CppCPU>();
  t.ToDevice(dev);
  EXPECT_NE(singa::defaultDevice, t.device());
}

TEST(TensorClass, CopyDataFromHostPtr) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);
  const float* dptr = static_cast<const float*>(t.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);
}

TEST(TensorClass, CopyData) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o(Shape{3});
  o.CopyData(t);
  const float* dptr = static_cast<const float*>(o.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);
}

TEST(TensorClass, Clone) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o = t.Clone();
  const float* dptr = static_cast<const float*>(o.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);
}

TEST(TensorClass, T) {
  Tensor t(Shape{2, 3});
  EXPECT_FALSE(t.transpose());
  Tensor o = t.T();  // o = t = {3,2}
  t.T();             // t = {2,3}
  EXPECT_EQ(true, o.transpose());
  EXPECT_EQ(t.block(), o.block());
  EXPECT_EQ(t.data_type(), o.data_type());
  EXPECT_EQ(t.shape()[0], o.shape()[1]);
  EXPECT_EQ(t.shape()[1], o.shape()[0]);
}

TEST(TensorClass, Repeat) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o = t.Repeat(vector<size_t>{2}, 9999);
  const float* dptr = static_cast<const float*>(o.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(1.0f, dptr[1]);
  EXPECT_FLOAT_EQ(2.0f, dptr[2]);
  EXPECT_FLOAT_EQ(2.0f, dptr[3]);
  EXPECT_FLOAT_EQ(3.0f, dptr[4]);
  EXPECT_FLOAT_EQ(3.0f, dptr[5]);
}

TEST(TensorClass, RepeatData) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o(Shape{6});
  o.RepeatData({2}, 9999, 2, t);
  const float* dptr = static_cast<const float*>(o.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(1.0f, dptr[1]);
  EXPECT_FLOAT_EQ(2.0f, dptr[2]);
  EXPECT_FLOAT_EQ(2.0f, dptr[3]);
  EXPECT_FLOAT_EQ(3.0f, dptr[4]);
  EXPECT_FLOAT_EQ(3.0f, dptr[5]);
}

TEST(TensorClass, Broadcast) {
  {
    Tensor a1(Shape{2, 3, 4, 5}), b1(Shape{5});
    auto c1 = Broadcast(a1, b1.shape()).shape();
    auto c2 = Broadcast(b1, a1.shape()).shape();
    EXPECT_EQ(c1[0], 2);
    EXPECT_EQ(c1[1], 3);
    EXPECT_EQ(c1[2], 4);
    EXPECT_EQ(c1[3], 5);

    EXPECT_EQ(c2[0], 2);
    EXPECT_EQ(c2[1], 3);
    EXPECT_EQ(c2[2], 4);
    EXPECT_EQ(c2[3], 5);
  }
  {
    Tensor a1(Shape{4, 5}), b1(Shape{2, 3, 4, 5});
    auto c1 = Broadcast(a1, b1.shape()).shape();
    auto c2 = Broadcast(b1, a1.shape()).shape();
    EXPECT_EQ(c1[0], 2);
    EXPECT_EQ(c1[1], 3);
    EXPECT_EQ(c1[2], 4);
    EXPECT_EQ(c1[3], 5);

    EXPECT_EQ(c2[0], 2);
    EXPECT_EQ(c2[1], 3);
    EXPECT_EQ(c2[2], 4);
    EXPECT_EQ(c2[3], 5);
  }
  {
    Tensor a1(Shape{1, 4, 5}), b1(Shape{2, 3, 1, 1});
    auto c1 = Broadcast(a1, b1.shape()).shape();
    auto c2 = Broadcast(b1, a1.shape()).shape();

    EXPECT_EQ(c1[0], 2);
    EXPECT_EQ(c1[1], 3);
    EXPECT_EQ(c1[2], 4);
    EXPECT_EQ(c1[3], 5);

    EXPECT_EQ(c2[0], 2);
    EXPECT_EQ(c2[1], 3);
    EXPECT_EQ(c2[2], 4);
    EXPECT_EQ(c2[3], 5);
  }
  {
    Tensor a1(Shape{3, 4, 5}), b1(Shape{2, 1, 1, 1});
    auto c1 = Broadcast(a1, b1.shape()).shape();
    auto c2 = Broadcast(b1, a1.shape()).shape();

    EXPECT_EQ(c1[0], 2);
    EXPECT_EQ(c1[1], 3);
    EXPECT_EQ(c1[2], 4);
    EXPECT_EQ(c1[3], 5);

    EXPECT_EQ(c2[0], 2);
    EXPECT_EQ(c2[1], 3);
    EXPECT_EQ(c2[2], 4);
    EXPECT_EQ(c2[3], 5);
  }
}
