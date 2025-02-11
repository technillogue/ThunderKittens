#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template<int M_BLOCK, int N_BLOCK>
struct batched_matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 2, 1, -1, -1, base_tile>; // Batch=2, K=16
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK][2], b[N_BLOCK][2]; }; // Batched input
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK][2]; }; // Batched output
    struct common_state   { coord<ducks::default_type> coord; }; // Use coord struct
    struct consumer_state { rt_fl<16, 64> accum[N_BLOCK][2]; }; // Batched accumulators
};

template<int _M_BLOCK=2, int _N_BLOCK=4>
struct batched_matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK;
    using layout    = batched_matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=2, PRODUCER_BARRIER_ARRIVALS=1; // Reduced stages to fix shared mem error
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K, int batch_size) {
        return dim3(PERISISTENT_GRID ? 132 : (M*N*batch_size)/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }

    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows / (M_BLOCK*64), Cblocks = args.globals.C.cols / (N_BLOCK*64);
        int batch_blocks = args.globals.C.depth; // Batch dimension is depth
        int task_id = args.task_iter*gridDim.x + blockIdx.x;

        if (task_id < batch_blocks * Rblocks * Cblocks) {
            int block_3d_id = task_id;
            int batch_idx = block_3d_id / (Rblocks * Cblocks);
            block_3d_id %= (Rblocks * Cblocks);
            int row_block_idx = block_3d_id / Cblocks;
            int col_block_idx = block_3d_id % Cblocks;
            args.common.coord = coord<ducks::default_type>{ row_block_idx * M_BLOCK, col_block_idx * N_BLOCK }; // Use coord constructor
            args.common.coord.d = batch_idx; // batch index as depth coord
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = 1; // Fixed K=16, so only 1 iter
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord.r = args.common.coord.dim<2>() + id; // Use dim<2> for row, dim<3> for col, dim<1> for depth, dim<0> for batch
        args.common.coord.c = args.common.coord.dim<3>();      // col block coord
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int batch_idx=0; batch_idx<2; ++batch_idx) { // Batch dimension loop
                    for(int i = 0; i < M_BLOCK; i++)
                        tma::load_async(args.input.a[i][batch_idx], args.globals.A,
                                        coord<ducks::default_type>{args.common.coord.dim<0>() + batch_idx, 0, args.common.coord.dim<2>()+i, args.iter}, args.inputs_arrived); // Use dim<0>, dim<2>
                    for(int i = 0; i < N_BLOCK; i++)
                        tma::load_async(args.input.b[i][batch_idx], args.globals.B,
                                        coord<ducks::default_type>{args.common.coord.dim<0>() + batch_idx, 0, args.iter, args.common.coord.dim<3>()+i}, args.inputs_arrived); // Use dim<0>, dim<3>
                }
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            for (int n = 0; n < N_BLOCK; n++) {
                zero(args.state.accum[n][0]); // Initialize for batch 0
                zero(args.state.accum[n][1]); // Initialize for batch 1
            }
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            for(int batch_idx=0; batch_idx<2; ++batch_idx) { // Batch dimension loop
                for(int n = 0; n < N_BLOCK; n++) {
                    warpgroup::mma_ABt(
                        args.state.accum[n][batch_idx],
                        args.input.a[warpgroup::groupid()][batch_idx],
                        args.input.b[n][batch_idx]
                    );
                }
            }
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            for(int batch_idx=0; batch_idx<2; ++batch_idx) { // Batch dimension loop
                for(int n = 0; n < N_BLOCK; n++) {
                    warpgroup::store(args.finish.c[warpgroup::groupid()][n][batch_idx], args.state.accum[n][batch_idx]);
                }
            }
            warpgroup::sync(warpgroup::groupid()+4);

            if(warpgroup::warpid() == 0) {
                for(int batch_idx=0; batch_idx<2; ++batch_idx) { // Batch dimension loop
                    for(int i = 0; i < N_BLOCK; i++) {
                        tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i][batch_idx],
                                       coord<ducks::default_type>{args.common.coord.dim<0>() + batch_idx, 0, args.common.coord.dim<2>(), args.common.coord.dim<3>()+i}); // Use dim<0>, dim<2>, dim<3>
                        tma::store_async_wait();
                    }
                }
            }

            // Zero the accumulators
            for (int n = 0; n < N_BLOCK; n++) {
                zero(args.state.accum[n][0]); // Initialize for batch 0
                zero(args.state.accum[n][1]); // Initialize for batch 1
            }
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};


constexpr bool NCU = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

// cpu_batched_gemm and run_benchmark functions remain the same as in the previous example.

template<typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t B, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using globals  = typename mmt::layout::globals;
    global_layout Ag{d_A, nullptr, nullptr, B, 1, M, K}; // Modified global layout init
    global_layout Bg{d_B, nullptr, nullptr, B, 1, K, N}; // Modified global layout init
    global_layout Cg{d_C, nullptr, nullptr, B, 1, M, N}; // Modified global layout init
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}


template<typename mmt>
int run_benchmark(size_t B, size_t M, size_t N, size_t K) {
    // ... [rest of the run_benchmark function remains the same] ...
    dim3 grid(mmt::grid(M, N, K, B));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    // ... [rest of the run_benchmark function remains the same] ...
}


int main() {
    int N, M, B, K;
    M = 4096;
    N = 4096;
    B = 2; // Batch size = 2
    K = 16; // K = 16 fixed
    run_benchmark<batched_matmul_template<2,4>>(B, M, N, K); // K = 16 fixed
    return 0;
}
