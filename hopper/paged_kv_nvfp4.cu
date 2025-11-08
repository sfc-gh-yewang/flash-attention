#include "paged_kv_nvfp4.h"

namespace flash {

__device__ __constant__ float custom_fp4_lut[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

} // namespace flash