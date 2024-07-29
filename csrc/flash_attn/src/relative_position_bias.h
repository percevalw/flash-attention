#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_causal, bool Has_alibi, bool Has_RPE>
struct RelativePositionBias {

    const float alibi_slope;
    const int max_seqlen_k, max_seqlen_q;

    __forceinline__ __device__ RelativePositionBias(const float alibi_slope, const int max_seqlen_k, const int max_seqlen_q)
        : alibi_slope(alibi_slope)
        , max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };


    template <typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void apply_relative_position_bias(
            Tensor1 &tensor,
            Tensor2 &sQP,
            const int col_idx_offset_,
            const int row_idx_offset,
            const int warp_row_stride,
            const int row_block_idx_offset
    ) {
        // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
        static_assert(Tensor1::layout_type::rank == 2, "Only support 2D Tensor");
        // static_assert(Layout::rank == 2, "Only support 2D Tensor");
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        if constexpr (Is_causal) {  // Simpler, we add the same bias vector to all rows
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(tensor); ++mi) {
                        tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                    }
                }
            }
        } else {  // Bias depends on both row_idx and col_idx
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                    const int row_idx = row_idx_base + i * 8;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;
                            if constexpr (Has_alibi) {
                                tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                            }
                            if constexpr (Has_RPE) {
                                const int diff_idx = std::min(127, std::max(0, col_idx - row_idx + 64));
                                const int block_row_idx = row_idx - row_block_idx_offset;
                                tensor(make_coord(i, mi), make_coord(j, nj)) += sQP(block_row_idx, diff_idx);
                            }
                        }
                    }
                }
            }
        }
    }

};

}  // namespace flash
