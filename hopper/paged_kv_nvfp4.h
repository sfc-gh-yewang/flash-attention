/******************************************************************************
 * NVFP4 paged loader that dequantizes to FP16/BF16 in registers and writes
 * the converted tile into shared memory (non-TMA path, SM90+).
 *
 * Layout assumptions (match your Python quantizer / nvfp4_to_fp16.cuh):
 *  - NVFP4: 2 values per byte (low nibble = val0, high nibble = val1)
 *  - Per-16-element scale, stored as FP8 e4m3 (1 byte), padded to a multiple
 *    of 4 scale-bytes per row (rounded_n_bytes).
 *  - Scales are indexed as: scale_idx = col // 16, with row-major layout:
 *      sf_offset = row * rounded_n_bytes + scale_idx
 *
 * The class below:
 *  - Loads NVFP4 from paged K/V global memory using the same page-table math
 *    as PagedKVManager.
 *  - Loads 1 scale byte per 16 elements (but we convert in 8-elt chunks;
 *    each 8-elt chunk picks the scale for its starting col).
 *  - Converts 8 NVFP4 values (packed in a uint32_t) -> 4 half2/bf162 in regs.
 *  - Writes those 8 FP16/BF16 elements into the smem tile (same layout that
 *    the mainloop expects for the non-TMA/cp.async path).
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include "cutlass/fast_math.h"
#include "utils.h"

// CUDA numeric helpers
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash {

using namespace cute;

// -------- FP4 helper (reuse the same LUT contract as nvfp4_to_fp16.cuh) -----
__constant__ float custom_fp4_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Convert a single 4-bit value (bit3=sign, bit[2:0]=magnitude-index) to float.
// We use the LUT for magnitude and OR-in the sign bit.
static inline __device__ float nvfp4_to_float(uint8_t v4) {
    float mag = custom_fp4_lut[v4 & 0x7];
    // inject sign into float bitpattern (sign bit at bit 31).
    return __int_as_float(__float_as_int(mag) | ((v4 & 0x8) << 28));
}

// Convert 8 NVFP4 values packed in a uint32_t into four float2's.
// Packing order per byte: (val2 << 4) | val1  (val1 = low nibble)
static inline __device__ void nvfp4x8_to_float2_vec(uint32_t pack, float2 (&out)[4]) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&pack);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint8_t pair = b[i];
        uint8_t v0 = (pair & 0x0F);
        uint8_t v1 = (pair >> 4);
        out[i].x = nvfp4_to_float(v0);
        out[i].y = nvfp4_to_float(v1);
    }
}

// Scale decode (FP8 e4m3)
static inline __device__ float decode_e4m3(uint8_t sf8) {
    __nv_fp8_e4m3 tmp;
    reinterpret_cast<uint8_t&>(tmp) = sf8;
    return float(tmp);
}

// Pack 8 scaled floats to FP16 (as 4 x 32-bit "half2" words).
static inline __device__ void float2x4_to_half2_u32(float2 (&f2)[4], float scale, uint32_t (&u32)[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 t;
        t.x = f2[i].x * scale;
        t.y = f2[i].y * scale;
        half2 h2 = __float22half2_rn(t);
        u32[i] = *reinterpret_cast<uint32_t*>(&h2);
    }
}

// Pack 8 scaled floats to BF16 (as 4 x 32-bit "bfloat162" words).
static inline __device__ void float2x4_to_bf162_u32(float2 (&f2)[4], float scale, uint32_t (&u32)[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 t;
        t.x = f2[i].x * scale;
        t.y = f2[i].y * scale;
        __nv_bfloat162 b2 = __float22bfloat162_rn(t);
        u32[i] = *reinterpret_cast<uint32_t*>(&b2);
    }
}

// ---------------------- PagedKVManagerFP4 (NVFP4 -> FP16/BF16) ----------------------

template <
    int kBlockN,                 // tile columns (N) per block
    int kHeadDim,                // K head dim (in elements)
    int kHeadDimV,               // V head dim (in elements)
    int NumThreads,              // producer threads
    typename ElementOut,         // FP16 or BF16 (cutlass::half_t or cutlass::bfloat16_t)
    bool  KV_Same_Iter = false,  // same-iter K/V streaming, like base manager
    int   LoadsPerRow_LB = 1     // same idea as base (lower bound on loads per row)
>
struct PagedKVManagerFP4 {
    // We keep most compile-time contracts from PagedKVManager so we can drop-in replace it.
    static constexpr bool SameHeadDim  = (kHeadDim == kHeadDimV);
    static constexpr int  kHeadDimGCD  = cute::gcd(kHeadDim, kHeadDimV);

    // We vectorize work along the "ElementOut" granularity on the smem side.
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(ElementOut); // 16B / 2B = 8 (for fp16/bf16)

    static_assert(kHeadDimGCD % LoadsPerRow_LB == 0, "Headdim and HeaddimV must be a multiple of LoadsPerRow_LB");
    static constexpr int kBytePerRowPacked = (kHeadDimGCD / 2 /* 2 elts/byte */) / LoadsPerRow_LB; // bytes of NVFP4 per row slice
    static constexpr int kBlockKSmem = ( ( (kHeadDimGCD/2) % 128 == 0 ? 128 : ((kHeadDimGCD/2) % 64 == 0 ? 64 : 32) ) / sizeof(ElementOut) ) * 2;
    // The 2 above is "elements per byte" for FP16/BF16 in smem vector stores (we ultimately store as ElementOut).

    // Page-table mechanics & tiling match the base manager.
    using ShapeKV       = cute::Shape<int32_t, int32_t, int32_t, int32_t>;   // (seqlen, d, head, batch)
    using StrideKV      = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapePageTable= cute::Shape<int32_t, int32_t>;                      // (batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;

    // Scale tensor shape: (seqlen, rounded_n_bytes, head, batch)  (rounded_n_bytes = ceil( (d/16) / 4 ) * 4 )
    using ShapeSF       = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideSF      = cute::Stride<int64_t, _1, int64_t, int64_t>;

    // For paging split: we reuse the "page offset + page id" trick from base.
    using TensorPageTable = decltype(make_tensor(make_gmem_ptr(static_cast<int const*>(nullptr)), ShapePageTable{}, StridePageTable{})(int(0), _));

    // NVFP4 global tensors (bytes)
    using TensorK4  = decltype(make_tensor(make_gmem_ptr(static_cast<uint8_t const*>(nullptr)), ShapeKV{}, StrideKV{})(_, _, int(0), _)); // d means bytes (d_bytes = d/2)
    using TensorV4  = TensorK4;

    // Per-16-element FP8 scale (bytes)
    using TensorSF  = decltype(make_tensor(make_gmem_ptr(static_cast<uint8_t const*>(nullptr)), ShapeSF{}, StrideSF{})(_, _, int(0), _));

    // Smem tiling on the destination (same partitioning as base manager uses for cp.async path).
    using GmemLayoutAtomKVCpAsync = Layout<Shape <Int<NumThreads / ( ( (kHeadDimGCD / sizeof(ElementOut)) % 64 == 0 ? 64 : 32) / (sizeof(cute::uint128_t)/sizeof(ElementOut)) )>,
                                                  Int< ( (kHeadDimGCD / sizeof(ElementOut)) % 64 == 0 ? 64 : 32) / (sizeof(cute::uint128_t)/sizeof(ElementOut)) >>,
                                          Stride<Int< ( (kHeadDimGCD / sizeof(ElementOut)) % 64 == 0 ? 64 : 32) / (sizeof(cute::uint128_t)/sizeof(ElementOut)) >, _1>>;
    // NOTE: We won’t actually copy with cp.async here; we just reuse the *same partitioning*
    //       so the smem layout the mainloop expects is identical.

    using GmemTiledCopyKVCpAsync = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementOut>{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));

    using GmemThrCopyKVCpAsync = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)));
    using TensortKcK = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{})));
    using TensortVcV = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{})));

    // Helper “page row index” bookkeeping per thread (same structure as base manager).
    static constexpr int kGmemThreadsPerRow = (sizeof(cute::uint128_t)/sizeof(ElementOut)) == 8
        ?  ( ( (kHeadDimGCD/2) % 128 == 0 ? 128 : ( (kHeadDimGCD/2) % 64 == 0 ? 64 : 32 ) ) / sizeof(cute::uint128_t) )
        : 8; // default safe
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");

    static constexpr int kPageEntryPerThread = cute::ceil_div(size<1>(TensortKcK{}), kGmemThreadsPerRow);
    using TensorPageOffset = decltype(make_tensor<cute::tuple<int, int>>(Shape<Int<kPageEntryPerThread>>{}));

    // Members (mirroring base)
    cutlass::FastDivmod const &page_size_divmod;
    cutlass::FastDivmod const &blockN_per_page_size_divmod;
    int const thread_idx;
    int const seqlen_k;
    int const leftpad_k;
    int const* const ptr_page_table;

    // Partition helpers
    GmemThrCopyKVCpAsync const gmem_thr_copy_kv;

    // Paged views
    TensorPageTable  mPageTable;
    TensorK4         mK_paged4;   // bytes: (seqlen, d_bytes=kHeadDim/2, head, batch)
    TensorV4         mV_paged4;   // bytes: (seqlen, d_bytes=kHeadDimV/2, head, batch)
    TensorSF         mK_sf;       // bytes: (seqlen, rounded_n_bytes, head, batch)
    TensorSF         mV_sf;       // bytes: (seqlen, rounded_n_bytes_v, head, batch)

    // Per-thread “which rows I own”
    TensorPageOffset tPrPageOffset;

    // For “TMA-like” paged indexing
    int bidb_kv_idx, bidb_kv_idx_prev, n_block_idx, n_block_idx_prev;

    CUTLASS_DEVICE
    PagedKVManagerFP4(
        int const* const ptr_page_table_,
        ShapePageTable  const& shape_pagetable,  StridePageTable const& stride_pagetable,
        uint8_t const*  ptr_K_fp4,               ShapeKV        const& shape_K,  StrideKV const& stride_K,
        uint8_t const*  ptr_V_fp4,               int const headdim_v, StrideKV const& stride_V,
        uint8_t const*  ptr_K_sf,                ShapeSF        const& shape_K_sf, StrideSF const& stride_K_sf,
        uint8_t const*  ptr_V_sf,                ShapeSF        const& shape_V_sf, StrideSF const& stride_V_sf,
        cutlass::FastDivmod const &page_size_divmod_,
        cutlass::FastDivmod const &blockN_per_page_size_divmod_,
        int const bidb, int const bidh, int const thread_idx_, int const seqlen_k_, int const leftpad_k_,
        int bidb_kv_idx_)
        : page_size_divmod(page_size_divmod_)
        , blockN_per_page_size_divmod(blockN_per_page_size_divmod_)
        , thread_idx(thread_idx_)
        , seqlen_k(seqlen_k_)
        , leftpad_k(leftpad_k_)
        , ptr_page_table(ptr_page_table_)
        , gmem_thr_copy_kv(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementOut>{},
                                           GmemLayoutAtomKVCpAsync{},
                                           Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}).get_thread_slice(thread_idx_))
        , bidb_kv_idx(bidb_kv_idx_)
        , bidb_kv_idx_prev(bidb_kv_idx_)
    {
        // Page table and paged tensors
        mPageTable = make_tensor(make_gmem_ptr(ptr_page_table), shape_pagetable, stride_pagetable)(bidb, _);

        // K/V NVFP4 bytes (second dimension is bytes: d_bytes = d/2)
        auto shape_K_bytes = make_shape(get<0>(shape_K), get<1>(shape_K) / 2, get<2>(shape_K), get<3>(shape_K));
        auto shape_V_bytes = make_shape(get<0>(shape_K), headdim_v / 2,       get<2>(shape_K), get<3>(shape_K));
        mK_paged4 = make_tensor(make_gmem_ptr(ptr_K_fp4), shape_K_bytes, stride_K)(_, _, bidh, _);
        mV_paged4 = make_tensor(make_gmem_ptr(ptr_V_fp4), shape_V_bytes, stride_V)(_, _, bidh, _);

        // Per-16-element scales (bytes), already padded per row to multiple of 4
        mK_sf = make_tensor(make_gmem_ptr(ptr_K_sf), shape_K_sf, stride_K_sf)(_, _, bidh, _);
        mV_sf = make_tensor(make_gmem_ptr(ptr_V_sf), shape_V_sf, stride_V_sf)(_, _, bidh, _);
    }

    // ----------------- Page-table helpers (same as base, but FP4 has no TMA) -----------------

    template <bool Seqlenk_mask=false, bool First_iter=false>
    CUTLASS_DEVICE void load_page_table(const int n_block) {
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            int const row = i * NumThreads
                          + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow)
                          + (thread_idx / kGmemThreadsPerRow);
            int const row_idx = n_block * kBlockN + row;
            int page_idx, page_off;
            page_idx = page_size_divmod.divmod(page_off, row_idx + leftpad_k);
            int const page = ((i + 1) * NumThreads <= kBlockN || row < kBlockN)
                             && (!Seqlenk_mask || row_idx < seqlen_k)
                             ? mPageTable[page_idx] : 0;
            tPrPageOffset[i] = {page, page_off};
        }
        // No V pointer carryover here; we recompute on demand.
        (void)First_iter;
    }

    // ----------------- Core: load & convert K (NVFP4 -> FP16/BF16 into smem) -----------------

    template <bool Seqlenk_mask=false, typename TensorKSmem>
    CUTLASS_DEVICE void load_K(const int n_block, TensorKSmem &&sK_pi /* position-independent smem tensor */) {
        // Same partitioning as base manager to index smem correctly
        auto thr = gmem_thr_copy_kv;
        Tensor tKsK = thr.partition_D(sK_pi); // ((Atom,AtomNum), ATOM_M, ATOM_N)
        Tensor cK   = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tKcK = thr.partition_S(cK);
        auto gmem_thr0 = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementOut>{},
                                         GmemLayoutAtomKVCpAsync{},
                                         Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}).get_thread_slice(_0{});
        Tensor t0KcK = gmem_thr0.partition_S(cK);

        // How many rows in this tile are valid for seqlen masking?
        int const seqlenk_row_limit = -int(get<0>(tKcK(_0{}, _0{}, _0{}))) + ( Seqlenk_mask
            ? min(seqlen_k - n_block * kBlockN, kBlockN)
            : kBlockN );

        // Iterate rows this thread owns
        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            bool const should_load_row = get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            if (!should_load_row) { continue; }

            // Base K row pointers in NVFP4 and SCALE space for this thread's paged rows
            auto [page, page_off] = tPrPageOffset[m / kGmemThreadsPerRow];
            // K row base in bytes (d_bytes is the second dim of mK_paged4)
            uint8_t const* k_row_bytes = &mK_paged4(page_off, _0{}, page);
            // Scale row base (rounded_n_bytes is 2nd dim of mK_sf)
            uint8_t const* k_sf_row    = &mK_sf(page_off, _0{}, page);

            // Vector along K (each step writes 8 FP16/BF16; reads 8 FP4 packed -> 4 bytes)
            #pragma unroll
            for (int k = 0; k < size<2>(tKsK); ++k) {
                // Column start (in elements)
                int col0 = get<1>(tKcK(_0{}, _0{}, k));
                // Load 8 NVFP4 from bytes: byte offset = col0 / 2
                int byte_off = col0 >> 1;
                uint32_t pack8 = *reinterpret_cast<const uint32_t*>(k_row_bytes + byte_off);

                // Pick per-16-element scale (scale index = col0 / 16).
                uint8_t sf8 = *(k_sf_row + (col0 >> 4));
                float    s  = decode_e4m3(sf8);

                // Convert to 4x u32 (each u32 packs 2 half/bf16)
                float2 f2[4]; nvfp4x8_to_float2_vec(pack8, f2);
                uint32_t u32[4];
                if constexpr (std::is_same<ElementOut, cutlass::bfloat16_t>::value) {
                    float2x4_to_bf162_u32(f2, s, u32);
                } else {
                    float2x4_to_half2_u32(f2, s, u32);
                }

                // Store 8 elements (=4x u32) into smem vector tKsK(:, m, k)
                // tKsK(_, m, k) is a contiguous vector of length kGmemElemsPerStore
                uint32_t* dst = reinterpret_cast<uint32_t*>(&tKsK(_0{}, m, k));
                #pragma unroll
                for (int j = 0; j < 4; ++j) { dst[j] = u32[j]; }
            }
        }
    }

    // ----------------- Core: load & convert V (NVFP4 -> FP16/BF16 into smem) -----------------

    template <bool Seqlenk_mask=false, typename TensorVSmem>
    CUTLASS_DEVICE void load_V(const int n_block, TensorVSmem &&sV_pi /* position-independent smem tensor */) {
        auto thr = gmem_thr_copy_kv;
        Tensor tVsV = thr.partition_D(sV_pi); // ((Atom,AtomNum), ATOM_M, ATOM_N)
        Tensor cV   = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});
        Tensor tVcV = thr.partition_S(cV);
        auto gmem_thr0 = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementOut>{},
                                         GmemLayoutAtomKVCpAsync{},
                                         Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}).get_thread_slice(_0{});
        Tensor t0VcV = gmem_thr0.partition_S(cV);

        int const seqlenk_row_limit = -int(get<0>(tVcV(_0{}, _0{}, _0{}))) + ( Seqlenk_mask
            ? min(seqlen_k - n_block * kBlockN, kBlockN)
            : kBlockN );

        #pragma unroll
        for (int m = 0; m < size<1>(tVsV); ++m) {
            // Guard for kBlockN not evenly divisible by the tiled copy
            if (m == size<1>(tVsV) - 1 && get<0>(tVcV(_0{}, m, _0{})) >= kBlockN) { break; }
            bool const should_load_row = get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
            if (!should_load_row) { continue; }

            // Paged row base for V (bytes) and its scales
            auto [page, page_off] = tPrPageOffset[m / kGmemThreadsPerRow];
            uint8_t const* v_row_bytes = &mV_paged4(page_off, _0{}, page);
            uint8_t const* v_sf_row    = &mV_sf(page_off, _0{}, page);

            #pragma unroll
            for (int k = 0; k < size<2>(tVsV); ++k) {
                int col0 = get<1>(tVcV(_0{}, _0{}, k));
                int byte_off = col0 >> 1;
                uint32_t pack8 = *reinterpret_cast<const uint32_t*>(v_row_bytes + byte_off);

                uint8_t sf8 = *(v_sf_row + (col0 >> 4));
                float    s  = decode_e4m3(sf8);

                float2 f2[4]; nvfp4x8_to_float2_vec(pack8, f2);
                uint32_t u32[4];
                if constexpr (std::is_same<ElementOut, cutlass::bfloat16_t>::value) {
                    float2x4_to_bf162_u32(f2, s, u32);
                } else {
                    float2x4_to_half2_u32(f2, s, u32);
                }

                uint32_t* dst = reinterpret_cast<uint32_t*>(&tVsV(_0{}, m, k));
                #pragma unroll
                for (int j = 0; j < 4; ++j) { dst[j] = u32[j]; }
            }
        }
    }

    // ----------------- PagedKV “store” path (AppendKV) in FP4 case -----------------
    // For AppendKV (writing back to global cache), if your runtime expects to keep
    // KV in NVFP4 on device, you’d mirror the conversion in reverse here. For now
    // (focus: decoding / read path), we don’t implement FP16->NVFP4 stores.
    template <typename TensorKSmem>
    CUTLASS_DEVICE void store_K(const int /*n_block*/, TensorKSmem&& /*ignored*/) { /* not implemented for FP4 */ }

    template <typename TensorVSmem>
    CUTLASS_DEVICE void store_V(const int /*n_block*/, TensorVSmem&& /*ignored*/) { /* not implemented for FP4 */ }
};

} // namespace flash
