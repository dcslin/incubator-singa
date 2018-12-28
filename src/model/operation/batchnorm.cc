#include "./batchnorm.h"

namespace singa {

BatchNormHandle::BatchNormHandle(const float momentum, const Tensor& input) {
  factor = momentum;
  batchsize = input.shape(0);
  channels = input.shape(1);
  if (input.nDim() == 4u) {
    height = input.shape().at(2);
    width = input.shape().at(3);
    is_2d = false;
  } else if (input.nDim() == 2u) {
    height = 1;
    width = 1;
    is_2d = true;
  } else {
    LOG(FATAL) << "The dimension of input should either be 4D or 2D.";
  }


#ifdef USE_MKLDNN
  dtype = GetMKLDNNDataType(input.data_type());
  epsilon =1e-5f;
  data_memory_format = is_2d ? mkldnn::memory::format::nc : mkldnn::memory::format::nchw;
  if (is_2d) {
    x_dims = {(int)batchsize, (int)channels} ;
    y_dims = {(int)batchsize, (int)channels} ;
  }
  else {
    x_dims = {(int)batchsize, (int)channels, (int)height, (int)width};
    y_dims = {(int)batchsize, (int)channels, (int)height, (int)width};
  }
  eng = new mkldnn::engine(mkldnn::engine::cpu, 0);
  x_md = new mkldnn::memory::desc(x_dims, dtype, data_memory_format);
  dx_md = new mkldnn::memory::desc(x_dims, dtype, data_memory_format);
  bn_fwd_d = new mkldnn::batch_normalization_forward::desc(mkldnn::forward_training, *x_md, epsilon,
                                                           mkldnn::use_scale_shift);
  bn_fwd_pd = new mkldnn::batch_normalization_forward::primitive_desc(*bn_fwd_d, *eng);
#endif // USE_MKLDNN

};


  BatchNormHandle::~BatchNormHandle() {
#ifdef USE_MKLDNN
    delete (eng);
    delete (x_md);
    delete (dx_md);
    delete (bn_fwd_d);
    delete (bn_fwd_pd);
#endif // USE_MKLDNN
  }

#ifdef USE_MKLDNN

  Tensor CpuBatchNormForwardInference(const BatchNormHandle &bnh, const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
                                                        Tensor& running_mean, Tensor& running_var){

    CHECK_EQ(x.device()->lang(), kCpp);
    Tensor y;
    y.ResetLike(x);


    Tensor w = get_bn_weight_from_scale_bias(bnScale, bnBias);

    y.device()->Exec([&y, &x, &w, &bnh](Context *ctx) {
      try {
        using namespace mkldnn;
        // examples of  mkldnn batch norm: https://github.com/intel/mkl-dnn/issues/367
        auto x_mem = memory({{{bnh.x_dims}, bnh.dtype, bnh.data_memory_format}, *bnh.eng}, x.block()->mutable_data());
        auto y_mem = memory({{{bnh.y_dims}, bnh.dtype, bnh.data_memory_format}, *bnh.eng}, y.block()->mutable_data());

        auto bn_fwd_d = batch_normalization_forward::desc(forward_inference, *bnh.x_md, bnh.epsilon, use_scale_shift);
        auto bn_fwd_pd = batch_normalization_forward::primitive_desc(bn_fwd_d, *bnh.eng);

        auto w_mem = memory(bn_fwd_pd.weights_primitive_desc(), w.block()->mutable_data());

        auto bn = batch_normalization_forward(bn_fwd_pd, x_mem, w_mem, y_mem);

        stream(stream::kind::eager).submit({bn}).wait();
      }
      catch (mkldnn::error &e) {
        InitLogging("");
        LOG(FATAL) << "MKLDNN Batch Norm" << "Status: " << e.status << " Message: " << e.message;
      }

    }, {x.block()}, {y.block()});

    return y;

  }

  const std::vector<Tensor>
  CpuBatchNormForwardTraining(const BatchNormHandle &bnh, const Tensor &x, const Tensor &bnScale, const Tensor &bnBias,
                              Tensor &running_mean, Tensor &running_var) {

    Tensor y;
    y.ResetLike(x);

    // mean and var for local batch
    Tensor mean;
    mean.ResetLike(running_mean);
    Tensor var;
    var.ResetLike(running_var);

    // combine scale and bias to construct weight tensor in required format for backward
    Tensor w = get_bn_weight_from_scale_bias(bnScale, bnBias);

    y.device()->Exec([&x, &y, &mean, &var, &w, &bnh](Context *ctx) {
      try {
        using namespace mkldnn;

        auto x_mem = memory({{{bnh.x_dims}, bnh.dtype, bnh.data_memory_format}, *bnh.eng},
                            x.block()->mutable_data());
        auto y_mem = memory({{{bnh.x_dims}, bnh.dtype, bnh.data_memory_format}, *bnh.eng},
                            y.block()->mutable_data());
        auto m_mem = memory(bnh.bn_fwd_pd->mean_primitive_desc(), mean.block()->mutable_data());

        auto v_mem = memory(bnh.bn_fwd_pd->variance_primitive_desc(), var.block()->mutable_data());

        auto w_mem = memory(bnh.bn_fwd_pd->weights_primitive_desc(),w.block()->mutable_data());

        auto bn_fwd = batch_normalization_forward(*bnh.bn_fwd_pd, x_mem, w_mem, y_mem, m_mem, v_mem);

        stream(stream::kind::eager).submit({bn_fwd}).wait();
      }
      catch (mkldnn::error &e) {
        singa::InitLogging("");
        LOG(FATAL) << "MKLDNN Batch Norm Backward" << "Status: " << e.status << " Message: " << e.message;
      }
    }, {x.block()}, {y.block(), mean.block(), var.block()});


    // local implemented running mean as mkldnn does not support it out of the box yet:
    // https://github.com/intel/mkl-dnn/issues/371
    running_mean = running_mean*bnh.factor + mean*(1-bnh.factor);
    running_var = running_var*bnh.factor + var*(1-bnh.factor);


    return {y, running_mean, running_var};

  }

const std::vector<Tensor> CpuBatchNormBackwardx(const BatchNormHandle &bnh,
                        const Tensor &y, const Tensor &dy,
                        const Tensor &x, //const Tensor &dx,
                        const Tensor &bnScale, const Tensor &bnBias,
                        const Tensor &mean, const Tensor &var){
  Tensor dx;
  dx.ResetLike(dy);

  // combine scale and bias to construct weight tensor in required format for backward
  Tensor w = get_bn_weight_from_scale_bias(bnScale, bnBias);

  Tensor dw(Shape{bnScale.Size(),bnBias.Size()});

  dx.device()->Exec([&dw, &x, &dx, &y, &dy, &w, &mean, &var, &bnh](Context *ctx) {

    try {
      using namespace mkldnn;

      auto  x_mem = memory({{{bnh.x_dims}, bnh.dtype, bnh.data_memory_format}, *bnh.eng},  x.block()->mutable_data());
      auto dx_mem = memory({{{bnh.x_dims}, bnh.dtype, bnh.data_memory_format}, *bnh.eng}, dx.block()->mutable_data());
      auto  y_mem = memory({{{bnh.x_dims}, bnh.dtype, bnh.data_memory_format}, *bnh.eng},  y.block()->mutable_data());
      auto dy_mem = memory({{{bnh.x_dims}, bnh.dtype, bnh.data_memory_format}, *bnh.eng}, dy.block()->mutable_data());

      auto m_mem = memory(bnh.bn_fwd_pd->mean_primitive_desc(), mean.block()->mutable_data());
      auto v_mem = memory(bnh.bn_fwd_pd->variance_primitive_desc(), var.block()->mutable_data());
      auto w_mem = memory(bnh.bn_fwd_pd->weights_primitive_desc(), w.block()->mutable_data());


      auto bn_bwd_d = batch_normalization_backward::desc(backward, *bnh.dx_md, *bnh.x_md, bnh.epsilon, use_scale_shift);
      auto bn_bwd_pd = batch_normalization_backward::primitive_desc(bn_bwd_d, *bnh.eng, *bnh.bn_fwd_pd);


      auto dw_mem = memory(bn_bwd_pd.diff_weights_primitive_desc());
      dw_mem.set_data_handle(dw.block()->mutable_data());

      auto bn_bwd = batch_normalization_backward(bn_bwd_pd, x_mem, m_mem, v_mem, dy_mem, w_mem, dx_mem, dw_mem);

      stream(stream::kind::eager).submit({bn_bwd}).wait();
    }
    catch (mkldnn::error &e) {
      singa::InitLogging("");
      LOG(FATAL) << "MKLDNN Batch Norm Backward" << "Status: " << e.status << " Message: " << e.message;
    }

  }, {x.block(), dy.block(), mean.block(), var.block()},
  {dx.block(), dw.block()});

  Tensor dbnScale = CopyRows(dw,0,bnScale.Size());
  Tensor dbnBias = CopyRows(dw,1,bnBias.Size());


  return {dx, dbnScale, dbnBias};
  }


#endif // USE_MKLDNN

#ifdef USE_CUDNN
CudnnBatchNormHandle::CudnnBatchNormHandle(const float momentum,
    const Tensor& input): BatchNormHandle(momentum, input) {
  if (is_2d)
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  else
    mode = CUDNN_BATCHNORM_SPATIAL;
  DataType dtype = input.data_type();
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&shape_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&param_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(shape_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype),
                                         batchsize,
                                         channels, height, width));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(param_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), 1, channels,
                                         1, 1));
};

const std::vector<Tensor> GpuBatchNormForwardTraining(const CudnnBatchNormHandle &cbnh,
                                   const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
                                   Tensor& running_mean, Tensor& running_var) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(bnScale.device()->lang(), kCuda);
  CHECK_EQ(bnBias.device()->lang(), kCuda);
  CHECK_EQ(running_mean.device()->lang(), kCuda);
  CHECK_EQ(running_var.device()->lang(), kCuda);

  Tensor mean, var;
  mean.ResetLike(running_mean);
  var.ResetLike(running_var);

  Shape shape = x.shape();

  Tensor input = x;  //for unification of 2d and 4d cases.
  if (cbnh.is_2d)
    input.Reshape(Shape{shape.at(0), shape.at(1), 1, 1});

  Tensor output;
  output.ResetLike(x);

  output.device()->Exec(
  [&](Context * ctx) {
    const float alpha = 1.0f, beta = 0.0f;
    double epsilon = CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
                  ctx->cudnn_handle, cbnh.mode, &alpha, &beta, cbnh.shape_desc,
                  input.block()->data(), cbnh.shape_desc, output.block()->mutable_data(),
                  cbnh.param_desc, bnScale.block()->data(), bnBias.block()->data(), cbnh.factor,
                  running_mean.block()->mutable_data(), running_var.block()->mutable_data(),
                  epsilon, mean.block()->mutable_data(),
                  var.block()->mutable_data()));
  },
  {input.block(), bnScale.block(), bnBias.block(), running_mean.block(), running_var.block()}, {
    output.block(), running_mean.block(), running_var.block(),
    mean.block(), var.block()
  });
  if (cbnh.is_2d) output.Reshape(Shape{shape.at(0), shape.at(1)});
  return {output, mean, var};
}

Tensor GpuBatchNormForwardInference(const CudnnBatchNormHandle &cbnh,
                                    const Tensor& x, const Tensor& bnScale,
                                    const Tensor& bnBias, const Tensor& running_mean, const Tensor& running_var) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(bnScale.device()->lang(), kCuda);
  CHECK_EQ(bnBias.device()->lang(), kCuda);
  CHECK_EQ(running_mean.device()->lang(), kCuda);
  CHECK_EQ(running_var.device()->lang(), kCuda);

  Shape shape = x.shape();

  Tensor input = x;  //for unification of 2d and 4d cases.
  if (cbnh.is_2d)
    input.Reshape(Shape{shape.at(0), shape.at(1), 1, 1});

  Tensor output;
  output.ResetLike(x);
  output.device()->Exec(
  [&](Context * ctx) {
    const float alpha = 1.0f, beta = 0.0f;
    double epsilon = CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
                  ctx->cudnn_handle, cbnh.mode, &alpha, &beta, cbnh.shape_desc,
                  input.block()->data(), cbnh.shape_desc, output.block()->mutable_data(),
                  cbnh.param_desc, bnScale.block()->data(), bnBias.block()->data(),
                  running_mean.block()->data(), running_var.block()->data(), epsilon));
  }, { input.block(), bnScale.block(), bnBias.block(), running_mean.block(), running_var.block() },
  {output.block()});
  return output;
}


const std::vector<Tensor> GpuBatchNormBackward(const CudnnBatchNormHandle &cbnh,
    const Tensor& dy, const Tensor& x, const Tensor& bnScale, const Tensor& mean,
    const Tensor& var) {
  CHECK_EQ(dy.device()->lang(), kCuda);
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(bnScale.device()->lang(), kCuda);
  CHECK_EQ(mean.device()->lang(), kCuda);
  CHECK_EQ(var.device()->lang(), kCuda);

  Tensor dx;
  dx.ResetLike(dy);

  Tensor dbnScale;
  dbnScale.ResetLike(bnScale);

  Tensor dbnBias;
  dbnBias.ResetLike(bnScale);

  dx.device()->Exec(
  [&](Context * ctx) {

    const float alpha = 1.0f, beta = .0f;
    double epsilon = CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationBackward(
                  ctx->cudnn_handle, cbnh.mode, &alpha, &beta, &alpha, &beta,
                  cbnh.shape_desc, x.block()->data(), cbnh.shape_desc, dy.block()->data(),
                  cbnh.shape_desc, dx.block()->mutable_data(), cbnh.param_desc,
                  bnScale.block()->data(), dbnScale.block()->mutable_data(),
                  dbnBias.block()->mutable_data(), epsilon, mean.block()->data(),
                  var.block()->data()));
  }, {x.block(), dy.block(), bnScale.block(), mean.block(), var.block()},
  {dx.block(), dbnScale.block(), dbnBias.block()});

  if (cbnh.is_2d) dx.Reshape(Shape{dx.shape().at(0), dx.shape().at(1)});

  return {dx, dbnScale, dbnBias};
}

#endif  //USE_CUDNN
}
