#ifndef SINGA_UTILS_MKLDNN_UTILS_H_
#define SINGA_UTILS_MKLDNN_UTILS_H_

#include <mkldnn.hpp>

#define MKL_CHECK(f)                                                               \
    do {                                                                       \
        mkldnn_status_t s = f;                                                 \
        if (s != mkldnn_success) {                                             \
            printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f,   \
                   s);                                                         \
            exit(2);                                                           \
        }                                                                      \
    } while (0)
#define CHECK_TRUE(expr)                                                       \
    do {                                                                       \
        int e_ = expr;                                                         \
        if (!e_) {                                                             \
            printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr);          \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

namespace singa {
  /*
   supported data type by mkldnn
   mkldnn_f32 - 32-bit/single-precision floating point.
   mkldnn_s32 - 32-bit signed integer.
   mkldnn_s16 - 16-bit signed integer.
   mkldnn_s8 - 8-bit signed integer.
   mkldnn_u8 - 8-bit unsigned integer.
   */
  inline mkldnn::memory::data_type GetMKLDNNDataType(DataType dtype) {
      mkldnn::memory::data_type ret = mkldnn::memory::data_type::f32;
      switch (dtype) {
          case kFloat32:
              ret = mkldnn::memory::data_type::f32;
          break;
          default:
              LOG(FATAL) << "The data type " << DataType_Name(dtype)
                         << " is not support by mkldnn";
      }
      return ret;
  }
}
#endif // SINGA_UTILS_MKLDNN_UTILS_H_
