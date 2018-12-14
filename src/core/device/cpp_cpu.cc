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

#include "singa/core/device.h"

// Required for posix_memalign
#define _POSIX_C_SOURCE 200112L
#ifdef _WIN32
#include <malloc.h>
#endif


namespace singa {

std::shared_ptr<Device> defaultDevice=std::make_shared<CppCPU>();

// malloc aligned memory chunk with alignment bytes
void *aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  void *p;
  return !posix_memalign(&p, alignment, size) ? p : NULL;
#endif
}
#ifdef _WIN32
void _free(void *ptr) {
  _aligned_free(ptr);
}
#else
void _free(void *ptr) {
    free(ptr);
  }
#endif

CppCPU::CppCPU() : Device(-1, 1) {
  lang_ = kCpp;
  // TODO(shicong): free engine, stream
#ifdef USE_MKLDNN
  MKL_CHECK(mkldnn_engine_create(&ctx_.engine, mkldnn_cpu, 0));
  MKL_CHECK(mkldnn_stream_create(&ctx_.stream, mkldnn_eager));
#endif //USE_MKLDNN
  //host_ = nullptr;
}

void CppCPU::SetRandSeed(unsigned seed) {
  ctx_.random_generator.seed(seed);
}


void CppCPU::DoExec(function<void(Context*)>&& fn, int executor) {
  CHECK_EQ(executor, 0);
  fn(&ctx_);
}


void* CppCPU::Malloc(int size) {
  if (size > 0) {
    void *ptr = aligned_malloc(size, 64);
    memset(ptr, 0, size);
    return ptr;
  } else {
    return nullptr;
  }
}


void CppCPU::Free(void* ptr) {
  if (ptr != nullptr)
    _free(ptr);
}


void CppCPU::CopyToFrom(void* dst, const void* src, size_t nBytes,
                           CopyDirection direction, Context* ctx) {
  memcpy(dst, src, nBytes);
}

}  // namespace singa
