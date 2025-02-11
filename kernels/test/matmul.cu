#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

constexpr int BATCH_SIZE = 4; // Added batch dimension

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    // Modified global layouts to include batch dimension
    using  global_layout  = gl<bf16, BATCH_SIZE, 1, -1, -1, base_tile>; // [B, 1, N, K]
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int batch; int2 coord; }; // Added batch to common state
    struct consumer_state { rt_fl<16, 64> accum[N_BLOCK]; };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERSISTENT_GRID=true> 
    __host__ static inline dim3 grid(int N, int M, int K) {
        const int blocks_per_batch = (N/(M_BLOCK*64)) * (M/(N_BLOCK*64));
        return dim3(PERSISTENT_GRID ? 132 : BATCH_SIZE * blocks_per_batch);
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        const int Rblocks = args.globals.C.rows / (M_BLOCK*64);
        const int Cblocks = args.globals.C.cols / (N_BLOCK*64);
        const int batches = BATCH_SIZE;
        const int total_batch_blocks = Rblocks * Cblocks;
        
        const int task_id = args.task_iter*gridDim.x + blockIdx.x;
        const int batch_idx = task_id / total_batch_blocks;
        const int batch_task = task_id % total_batch_blocks;

        if (task_id >= batches * total_batch_blocks) {
            args.num_iters = -1;
            return;
        }

        // Original coordinate calculation within batch
        const int super_rows = (Rblocks/SUPER_M)*SUPER_M;
        const int final_rows = Rblocks - super_rows;
        const int super_repeat = SUPER_M*Cblocks;

        if (batch_task < super_rows * Cblocks) {
            args.common.coord = {SUPER_M*(batch_task/super_repeat) + batch_task%SUPER_M, 
                                (batch_task%super_repeat)/SUPER_M};
        }
        else if (batch_task < Rblocks*Cblocks) {
            const int remainder_id = batch_task - super_rows*Cblocks;
            args.common.coord = {super_rows + (remainder_id%final_rows), 
                                remainder_id/final_rows};
        }
        else {
            args.num_iters = -1;
            return;
        }

        args.common.batch = batch_idx;
        args.num_iters = args.globals.A.cols/64;  // K dimension
        const int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = {args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK};
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }

        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                // Modified TMA loads with batch index
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                   {args.common.batch, 0, args.common.coord.x+i, args.iter}, 
                                   args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                   {args.common.batch, 0, args.iter, args.common.coord.y+i}, 
                                   args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            for (int n = 0; n < N_BLOCK; n++) 
                zero(args.state.accum[n]);
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            for(int n = 0; n < N_BLOCK; n++) {
                warpgroup::mma_ABt(args.state.accum[n], args.input.a[warpgroup::groupid()], args.input.b[n]);
            }
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }

        __device__ static void finish(consumer_finish_args<layout> args) {
            for(int n = 0; n < N_BLOCK; n++) {
                warpgroup::store(args.finish.c[warpgroup::groupid()][n], args.state.accum[n]);
            }
            warpgroup::sync(warpgroup::groupid()+4);
            
            if(warpgroup::warpid() == 0) {
                for(int i = 0; i < N_BLOCK; i++) {
                    // Modified TMA store with batch index
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                   {args.common.batch, 0, args.common.coord.x, args.common.coord.y+i});
                    tma::store_async_read_wait();
                }
            }

            for(int n = 0; n < N_BLOCK; n++) zero(args.state.accum[n]);
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};


constexpr bool NCU = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int B, int M, int N, int K) {
    #pragma omp parallel for collapse(3) // Parallelize over batches and matrices
    for (int b_idx = 0; b_idx < B; ++b_idx) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a[b_idx * M * K + i * K + k] * b[b_idx * K * N + k * N + j];
                }
                c[b_idx * M * N + i * N + j] = sum;
            }
        }
    }
}

template<typename mmt>
int run_benchmark(int B, int N, int M, int K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  B=" << B << " N=" << N << " M=" << M << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << mmt::M_BLOCK*64 << "x" << mmt::N_BLOCK*64 << "\n";

    // Host allocations with batch dimension
    float *h_A = new float[B * N * K];
    float *h_B = new float[B * K * M];
    float *h_C = new float[B * N * M];
    float *h_C_ref = new float[B * N * M];

    // Initialize matrices
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N*K; ++i) h_A[b*N*K + i] = dis(gen);
        for (int i = 0; i < K*M; ++i) h_B[b*K*M + i] = dis(gen);
    }

    // CPU reference
    cpu_gemm(h_A, h_B, h_C_ref, B, N, M, K);

    // Device allocations
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, B*N*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, B*K*M*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, B*N*M*sizeof(__nv_bfloat16));

    // Convert and copy data
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[B*N*K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[B*K*M];
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N*K; ++i) h_A_bf16[b*N*K + i] = __float2bfloat16(h_A[b*N*K + i]);
        for (int i = 0; i < K*M; ++i) h_B_bf16[b*K*M + i] = __float2bfloat16(h_B[b*K*M + i]);
    }
    cudaMemcpy(d_A, h_A_bf16, B*N*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, B*K*M*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Configure kernel
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    dim3 grid = mmt::grid(N, M, K);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    std::cout << "Launching kernel with grid (" << grid.x << " blocks)" << std::endl;

    // Warmup
    for(int i = 0; i < (NCU ? 0 : 2); i++) {
        prototype::lcf::kernel<mmt><<<grid, block, mem_size>>>({d_A, d_B, d_C});
    }

    // Timing
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    const int ITERS = 10;
    for(int i = 0; i < ITERS; i++) {
        prototype::lcf::kernel<mmt><<<grid, block, mem_size>>>({d_A, d_B, d_C});
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double tflops = (2.0 * B * N * M * K * ITERS) / (diff.count() * 1e12);

    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // Verify results
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[B*N*M];
    cudaMemcpy(h_C_bf16, d_C, B*N*M*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Convert to float and check
    int total_errors = 0;
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N*M; ++i) {
            float val = __bfloat162float(h_C_bf16[b*N*M + i]);
            float ref = h_C_ref[b*N*M + i];
            if (fabs(val - ref) > 0.1) { // Adjusted tolerance for bfloat16
                if (total_errors < 10) 
                    std::cerr << "Batch " << b << " Error at (" << i/M << "," << i%M 
                              << "): " << val << " vs " << ref << std::endl;
                total_errors++;
            }
        }
    }

    std::cout << "Total errors: " << total_errors << "/" << B*N*M << std::endl;

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    delete[] h_A_bf16; delete[] h_B_bf16; delete[] h_C_bf16;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return total_errors > 0 ? 1 : 0;
}

int main() {
    constexpr int BATCH_SIZE = 4;
    constexpr int K = 16; // Must be >=64 for current implementation
    
    // Test with different sizes
    int sizes[] = {3072, 12288};
    for (int size : sizes) {
        int N = size, M = size;
        if (run_benchmark<matmul_template<2,4,8>>(BATCH_SIZE, N, M, K) != 0) {
            std::cerr << "Validation failed for size " << size << std::endl;
            return 1;
        }
    }
    return 0;
}

