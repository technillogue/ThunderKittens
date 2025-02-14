#include "kittens.cuh"
#include "prototype.cuh"
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>


#define CUDA_CHECK(cond)                                                   \
  do {                                                                     \
    cudaError_t err = cond;                                                \
    if (err != cudaSuccess)                                                \
    {                                                                      \
      std::cerr << "[CUDA_CHECK] Cuda error: " << cudaGetErrorString(err) << std::endl; \
      std::cerr << "code: " << #cond << std::endl;                         \
      exit(1);                                                             \
    }                                                                      \
  } while (false)

int getenv(const char* name, int default_value) {
  auto value = std::getenv(name);
  return (value == nullptr || value[0] == '\0') ? default_value : std::stoi(value);
}

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

constexpr int BATCH_SIZE = 4;

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, -1, 1, -1, -1, base_tile>; // [B, 1, N, K]
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int batch; int2 coord; };
    struct consumer_state { rt_fl<16, 64> accum[N_BLOCK]; };
};



template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=8> // Reduced SUPER_M for better task distribution
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;

    template<bool PERSISTENT_GRID=false>
    __host__ static inline dim3 grid(int N, int M, int K) {
        const int blocks_per_batch = ((N + M_BLOCK*64 - 1)/(M_BLOCK*64)) * 
                                   ((M + N_BLOCK*64 - 1)/(N_BLOCK*64));
        const int total_blocks = BATCH_SIZE * blocks_per_batch;
        std::cout << total_blocks << std::endl;
        return dim3(std::min(total_blocks, 108)); // Cap grid size for H100 SMs
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        printf("common setup\n");
        const int Rblocks = (args.globals.C.rows + M_BLOCK*64 - 1) / (M_BLOCK*64);
        const int Cblocks = (args.globals.C.cols + N_BLOCK*64 - 1) / (N_BLOCK*64);
        const int total_blocks_per_batch = Rblocks * Cblocks;
        
        const int task_id = blockIdx.x + gridDim.x * args.task_iter;
        const int batch_idx = task_id / total_blocks_per_batch;
        const int batch_task = task_id % total_blocks_per_batch;

        if (batch_idx >= BATCH_SIZE) {
            args.num_iters = -1;
            return;
        }

        // Calculate matrix block coordinates
        const int matrix_row = (batch_task / Cblocks) * M_BLOCK;
        const int matrix_col = (batch_task % Cblocks) * N_BLOCK;
        
        args.common.batch = batch_idx;
        args.num_iters = (args.globals.A.cols + 63) / 64;
        args.common.coord = {matrix_row, matrix_col};
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
            printf("prducer setup\n");
        }

        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                printf("tma::expect arrive\n");
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++) {
                    const int row = args.common.coord.x + i;
                    if(row * 64 < args.globals.A.rows) {
                        tma::load_async(args.input.a[i], args.globals.A,
                                       {args.common.batch, 0, row, args.iter},
                                       args.inputs_arrived);
                    }
                }
                for(int i = 0; i < N_BLOCK; i++) {
                    const int col = args.common.coord.y + i;
                    if(col * 64 < args.globals.B.cols) {
                        tma::load_async(args.input.b[i], args.globals.B,
                                       {args.common.batch, 0, args.iter, col},
                                       args.inputs_arrived);
                    }
                }
            }
        }

    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            for (int n = 0; n < N_BLOCK; n++) 
                zero(args.state.accum[n]);
            printf("consumer setup done\n");
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            for(int n = 0; n < N_BLOCK; n++) {
                if(args.common.coord.y + n*64 < args.globals.C.cols) {
                    warpgroup::mma_ABt(args.state.accum[n], 
                                     args.input.a[warpgroup::groupid()], 
                                     args.input.b[n]);
                }
            }
            printf("compute async wait lane: %d\n", laneid());
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }

        __device__ static void finish(consumer_finish_args<layout> args) {
            const int base_row = args.common.coord.x + warpgroup::groupid()*64;
            
            for(int n = 0; n < N_BLOCK; n++) {
                const int base_col = args.common.coord.y + n*64;
                if(base_row < args.globals.C.rows && base_col < args.globals.C.cols) {
                    warpgroup::store(args.finish.c[warpgroup::groupid()][n], 
                                   args.state.accum[n]);
                }
            }
            printf("finish sync\n");
            warpgroup::sync(0); // Use unified barrier
            
            if(warpgroup::warpid() == 0 && laneid() == 0) {
                for(int i = 0; i < N_BLOCK; i++) {
                    const int col = args.common.coord.y + i*64;
                    if(col < args.globals.C.cols) {
                        tma::store_async(args.globals.C, 
                                       args.finish.c[warpgroup::groupid()][i],
                                       {args.common.batch, 0, 
                                        args.common.coord.x, 
                                        col});
                    printf("store async read wait i: %d", i);
                    tma::store_async_read_wait();

                    }
                }
                // tma::store_async_commit();
            }

            for(int n = 0; n < N_BLOCK; n++) zero(args.state.accum[n]);
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};


bool NCU = getenv("NCU", 0) != 0;

void cpu_gemm(float* a, float* b, float* c, int B, int N, int M, int K) {
    #pragma omp parallel for collapse(3)
    for (int b_idx = 0; b_idx < B; b_idx++) {
        for (int n_idx = 0; n_idx < N; n_idx++) {
            for (int m_idx = 0; m_idx < M; m_idx++) {
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < K; k_idx++) {
                    sum += a[(b_idx * N * K) + (n_idx * K) + k_idx] * 
                           b[(b_idx * K * M) + (k_idx * M) + m_idx];
                }
                c[(b_idx * N * M) + (n_idx * M) + m_idx] = sum;
            }
        }
    }
}

template<typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, int B, int N, int M, int K, dim3 grid, dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using globals  = typename mmt::layout::globals;
    
    global_layout Ag{d_A, B, nullptr, N, K};
    global_layout Bg{d_B, B, nullptr, K, M};
    global_layout Cg{d_C, B, nullptr, N, M};
    globals G{Ag, Bg, Cg};
    std::cout << "yahoo\n";

    
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

template<typename mmt>
int run_benchmark(int N, int M, int K) {
    const int B = BATCH_SIZE;

    std::cout << "----- Batch Matrix Multiply [B=" << B << ", N=" << N << ", M=" << M << ", K=" << K << "] -----\n";

    // Host allocations
    float *h_A = new float[B*N*K];
    float *h_B = new float[B*K*M];
    float *h_C = new float[B*N*M];
    float *h_C_ref = new float[B*N*M];

    // Initialize matrices
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);
    for(int i = 0; i < B*N*K; ++i) h_A[i] = dis(gen);
    for(int i = 0; i < B*K*M; ++i) h_B[i] = dis(gen);

    // CPU reference
    cpu_gemm(h_A, h_B, h_C_ref, B, N, M, K);
    std::cout << "cpu done\n";

    // Device allocations
    bf16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, B*N*K*sizeof(bf16));
    cudaMalloc(&d_B, B*K*M*sizeof(bf16));
    cudaMalloc(&d_C, B*N*M*sizeof(bf16));

    // Convert and copy data
    bf16 *h_A_bf16 = new bf16[B*N*K];
    bf16 *h_B_bf16 = new bf16[B*K*M];
    for(int i = 0; i < B*N*K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for(int i = 0; i < B*K*M; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    
    cudaMemcpy(d_A, h_A_bf16, B*N*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, B*K*M*2, cudaMemcpyHostToDevice);

    std::cout << "memcpy done\n";

    // Launch kernel
    dim3 grid = mmt::grid(N, M, K);
    dim3 block(kittens::WARP_THREADS * (mmt::NUM_CONSUMER_WARPS + 1));
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY-1024);

    // Warmup
    for(int i = 0; i < (NCU ? 0 : 2); i++) {
        inner_run<mmt>(d_A, d_B, d_C, B, N, M, K, grid, block);
    }
    // manually check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in kernel launch: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }

    std::cout << "warmup\n";
    
    // Timing
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++) {
        inner_run<mmt>(d_A, d_B, d_C, B, N, M, K, grid, block);
    }
    cudaDeviceSynchronize();
    std::cout << "timing\n";
    auto end = std::chrono::high_resolution_clock::now();

    // Copy back and verify
    bf16 *h_C_bf16 = new bf16[B*N*M];
    cudaMemcpy(h_C_bf16, d_C, B*N*M*2, cudaMemcpyDeviceToHost);
    for(int i = 0; i < B*N*M; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    // Validation
    float max_error = 0.0f;
    for(int b = 0; b < B; b++) {
        for(int i = 0; i < N*M; i++) {
            float error = fabs(h_C[b*N*M + i] - h_C_ref[b*N*M + i]);
            max_error = fmaxf(max_error, error);
        }
    }
    std::cout << "Max error: " << max_error << "\n";

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    delete[] h_A_bf16; delete[] h_B_bf16; delete[] h_C_bf16;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}

int main() {
    // Test with different sizes
    int device = getenv("DEVICE", 0);
    CUDA_CHECK(cudaSetDevice(device));

    run_benchmark<matmul_template<2,4,8>>(64, 64, 64);
    // run_benchmark<matmul_template<2,4,8>>(12288, 12288, 16);
    return 0;
}
