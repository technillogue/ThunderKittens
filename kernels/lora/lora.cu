#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include "kittens.cuh"               // Or wherever your TK root header is
#include "prototype.cuh"            // For the lcf template
#include "ops/group/wgmma/wgmma.cuh"// For wgmma calls if needed
#include "types/types.cuh"          // For bf16, etc.

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// ---------------------------------------------------------------------------
// 1. Define a layout describing how we store A, B, and C in shared memory
//    for each step, plus the ephemeral "common_state" and "consumer_state".
// ---------------------------------------------------------------------------
template<int B_MAX=2> 
struct lora_layout
{
    // We'll do a tile of A as st_bf<64, 32> for wgmma 64x64x32, if we want 32 as the K-tile chunk.
    // On H100 with BF16, 64x64x32 is a common wgmma shape.
    // We'll store up to B=2, so we define an array of these for each b.
    using tileA = st_bf<64, 32>;  
    using tileB = st_bf<32, 64>;  
    using tileC = st_bf<64, 64>;  // final tile of dimension 64x64

    // We'll define global layouts for up to B=2, each a standard 2D BF16 layout:
    // gl<bf16, -1, -1> means we have 2 runtime dims (the row & col).
    // But we only store "one 2D matrix" in each. We'll store them in an array or struct with B=2.

    struct globals {
      // Pointers to A[0], B[0], A[1], B[1], plus a pointer to C. We can keep them separate.
      bf16 *A0, *A1;  // shape [N, K]
      bf16 *B0, *B1;  // shape [K, M]
      bf16 *C;        // shape [N, M]
      int N, K, M;    // We'll treat B up to 2 with either A1/B1 == nullptr or not
      int B;          // how many loras? (0,1,2)
    };

    // "input_block" in LCF is the portion of shared memory that producers fill:
    // We'll store tileA/tileB for b=0 or b=1. 
    struct input_block {
      tileA A[B_MAX];
      tileB B[B_MAX];
    };

    // "finish_block" is the portion of shared memory that consumer uses to store final tile
    struct finish_block {
      tileC C;  // one 64×64 chunk for the final output
    };

    // "common_state" is ephemeral state that is shared among producers & consumers.
    struct common_state {
      int tileN_idx, tileM_idx; // which 64×64 block in the NxM space
      int k_chunks;             // number of 32-chunk steps to cover K
    };

    // "consumer_state" is private to the consumer warps: typically the register tiles
    struct consumer_state {
      // We'll keep an accumulator tile for the final result. We can do 64x64 in float
      // and accumulate multiple b's, multiple k-chunks, etc.
      rt_fl<64,64> accum;  
    };
};

// ---------------------------------------------------------------------------
// 2. The "template" that LCF uses to do the Producer/Consumer steps
//    We'll define how many warps are producers vs consumers, etc.
// ---------------------------------------------------------------------------
template<int B_MAX=2>
struct lora_matmul {
  using layout = lora_layout<B_MAX>;
  static constexpr int NUM_CONSUMER_WARPS = 4;  // # warps for compute
  static constexpr int INPUT_PIPE_STAGES  = 2;  // pipeline depth
  static constexpr int PRODUCER_BARRIER_ARRIVALS = 1; // typical

  // Helper: figure out how big the grid is 
  // We'll do a persistent kernel so gridDim.x can be e.g. 132 or so.
  // The kernel itself loops over all tileN * tileM. We only do 1D grid here, ignoring blockIdx.y.
  __host__ static dim3 grid(int N, int M) {
    // We'll pick something around 128 or so for large concurrency
    return dim3(132,1,1);
  }

  // LCF calls "common_setup" once per block, per iteration:
  // We compute which "tile of (N×M) space" we handle, or if done, set num_iters=-1 to stop.
  __device__ static void common_setup(common_setup_args<layout> args)
  {
    auto &G = args.globals; 
    int total_N_tiles = (G.N + 63) / 64;
    int total_M_tiles = (G.M + 63) / 64;
    int tile_id = args.task_iter*gridDim.x + blockIdx.x;
    if(tile_id >= total_N_tiles * total_M_tiles) {
      // no more work
      args.num_iters = -1;
      return;
    }

    // decode the tile id => (tileN_idx, tileM_idx)
    int tileM_idx = tile_id / total_N_tiles;
    int tileN_idx = tile_id % total_N_tiles;
    
    args.common.tileN_idx = tileN_idx;
    args.common.tileM_idx = tileM_idx;

    // K-chunks of size 32 each
    int k_chunks = (G.K + 31)/32;  // how many 32-wide slices of K
    args.common.k_chunks = k_chunks;

    // The total number of iterations for the pipeline is "k_chunks"
    // We'll do one iteration per chunk. 
    args.num_iters = k_chunks;
  }

  // Producer struct: loads A, B from global memory to shared, in slices of K=32
  struct producer {
    __device__ static void setup(producer_setup_args<layout>) {
      // We can do e.g. warpgroup::decrease_registers<32>() if we want to free registers
    }

    __device__ static void load(producer_load_args<layout> args)
    {
      // producer warps fill args.input.(A[0],B[0],A[1],B[1]) from global memory
      // depending on how many B we have in args.globals.B
      auto &G = args.globals;
      auto &inp = args.input;

      int chunk_idx = args.iter;  // which K-chunk
      int k_begin = chunk_idx * 32;
      int k_size  = min(32, G.K - k_begin);

      // We'll do tma::load_async if you prefer TMA, or just a simpler warp-group global->shared copy.
      // For brevity here, we do a naive "warpgroup::load" if it exists. 
      // But TMA is recommended for H100. For example:
      //   tma::load_async(..., G.A0, { row_tile, k_begin }, args.inputs_arrived );
      // Below is a simplified version. We skip partial tail for k_size < 32, but let's assume K is multiple of 32 for now.

      // tile offsets:
      int tileN_offset = args.common.tileN_idx*64;
      int tileM_offset = args.common.tileM_idx*64;

      // We'll load sub-block from A0 => st_bf<64,32> in shared
      // coordinate in A: row from tileN_offset..tileN_offset+63, col from k_begin..k_begin+31
      // (We have to clamp for partial edges if N not multiple of 64, but let's skip that for the example.)
      
      if (G.B > 0) {
        // load A0 tile:
        warpgroup::group<2>::load_async(inp.A[0], G.A0, { tileN_offset, k_begin }, args.inputs_arrived);
        // load B0 tile:
        warpgroup::group<2>::load_async(inp.B[0], G.B0, { k_begin, tileM_offset }, args.inputs_arrived);
      }

      if (G.B > 1) {
        // load A1 tile:
        warpgroup::group<2>::load_async(inp.A[1], G.A1, { tileN_offset, k_begin }, args.inputs_arrived);
        // load B1 tile:
        warpgroup::group<2>::load_async(inp.B[1], G.B1, { k_begin, tileM_offset }, args.inputs_arrived);
      }
    }
  };

  // Consumer struct: does the matmul in register tiles, accumulates partial sums, and outputs
  struct consumer {
    __device__ static void setup(consumer_setup_args<layout> args)
    {
      // allocate more registers for the consumer warps
      // warpgroup::increase_registers<192>();
      zero(args.state.accum);
    }

    __device__ static void compute(consumer_compute_args<layout> args)
    {
      // We have up to 2 sets of (A,B) tiles in shared memory. 
      // We'll do wgmma 64x64x32 on each pair, accumulate in register tile `args.state.accum`.
      // For BFS, we typically do `warpgroup::mma_AB( accum, inputA, inputB )`, etc.

      // wait for producers:
      warpgroup::mma_async_wait();  

      // For B=1 or B=2, do it:
      auto &inp = args.input;
      auto &G   = args.globals;
      if (G.B > 0) {
        warpgroup::mma_AB(args.state.accum, inp.A[0], inp.B[0]);
      }
      if (G.B > 1) {
        warpgroup::mma_AB(args.state.accum, inp.A[1], inp.B[1]);
      }

      // arrive at barrier so producers know they can move on:
      if(threadIdx.x % warpSize == 0) arrive(args.inputs_finished);
    }

    __device__ static void finish(consumer_finish_args<layout> args)
    {
      // After all k-chunks done, we store the final tile to global memory
      // We'll store to st_bf<64,64> in shared memory, then TMA store out to global

      // 1) store accum => shared
      warpgroup::store(args.finish.C, args.state.accum);
      warpgroup::sync(warpgroup::groupid()+4);

      // 2) then store from shared => G.C
      auto &G = args.globals;
      int tileN_offset = args.common.tileN_idx*64;
      int tileM_offset = args.common.tileM_idx*64;

      // TMA or normal copy. For now let's do a simpler store:
      warpgroup::group<2>::store_async(G.C, args.finish.C, { tileN_offset, tileM_offset });
      tma::store_async_wait<0>();  // wait for store to complete

      // reset accum
      zero(args.state.accum);

      // final arrive
      if(threadIdx.x % warpSize == 0) arrive(args.finish_finished);
    }
  };
}; // end struct lora_matmul


// ---------------------------------------------------------------------------
// 3. A small host "test" harness for demonstration
// ---------------------------------------------------------------------------
__host__ void run_lora_matmul(
    __nv_bfloat16* A0, __nv_bfloat16* B0,
    __nv_bfloat16* A1, __nv_bfloat16* B1,
    __nv_bfloat16* C,
    int N, int K, int M, int B) 
{
  using mmt = lora_matmul<2>;
  using GL = mmt::layout::globals;

  // build a "globals" struct
  GL G;
  G.A0 = A0; 
  G.A1 = A1; 
  G.B0 = B0; 
  G.B1 = B1;
  G.C  = C;
  G.N = N;
  G.K = K;
  G.M = M;
  G.B = B; // how many loras we have active

  // pick block size. LCF expects a typical block of 256 or 512 threads
  // each block has (producer warps + consumer warps), let's do 8 warps => 256 threads
  dim3 block(256,1,1);
  // pick a grid
  dim3 grid = mmt::grid(N, M);

  // dynamic shared mem for LCF pipeline
  // Typically we want large, e.g. 96 KB minus some overhead
  size_t shmem_size = 96*1024;

  // set attributes for max dynamic smem
  cudaFuncSetAttribute(
    kittens::prototype::lcf::kernel<mmt>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    (int)shmem_size
  );

  // launch
  kittens::prototype::lcf::kernel<mmt><<<grid, block, shmem_size>>>(G);
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// 4. (Optional) A main() showing a minimal usage
// ---------------------------------------------------------------------------
int main()
{
  int N=4096, K=256, M=4096;
  int B=2; // up to 2 loras
  // allocate host memory
  size_t szA = N*K, szB = K*M, szC = N*M;
  __nv_bfloat16* hA0  = new __nv_bfloat16[szA];
  __nv_bfloat16* hB0  = new __nv_bfloat16[szB];
  __nv_bfloat16* hA1  = new __nv_bfloat16[szA];
  __nv_bfloat16* hB1  = new __nv_bfloat16[szB];
  __nv_bfloat16* hC   = new __nv_bfloat16[szC];

  // init random
  for(size_t i=0; i<szA; i++){
    float val = (float) (rand()%1000 - 500)*0.001f;
    hA0[i] = __float2bfloat16(val);
    hA1[i] = __float2bfloat16(val*0.5f);
  }
  for(size_t i=0; i<szB; i++){
    float val = (float) (rand()%1000 - 500)*0.001f;
    hB0[i] = __float2bfloat16(val);
    hB1[i] = __float2bfloat16(val*0.1f);
  }

  // allocate device
  __nv_bfloat16 *dA0, *dB0, *dA1, *dB1, *dC;
  cudaMalloc(&dA0, szA*sizeof(__nv_bfloat16));
  cudaMalloc(&dB0, szB*sizeof(__nv_bfloat16));
  cudaMalloc(&dA1, szA*sizeof(__nv_bfloat16));
  cudaMalloc(&dB1, szB*sizeof(__nv_bfloat16));
  cudaMalloc(&dC,  szC*sizeof(__nv_bfloat16));

  // copy up
  cudaMemcpy(dA0, hA0, szA*2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB0, hB0, szB*2, cudaMemcpyHostToDevice);
  cudaMemcpy(dA1, hA1, szA*2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB1, hB1, szB*2, cudaMemcpyHostToDevice);

  // run kernel
  run_lora_matmul(dA0, dB0, dA1, dB1, dC, N, K, M, B);

  // copy result back
  cudaMemcpy(hC, dC, szC*2, cudaMemcpyDeviceToHost);

  std::cout << "Done with batch_lora_lcf!\n";

  // Cleanup 
  cudaFree(dA0);
  cudaFree(dB0);
  cudaFree(dA1);
  cudaFree(dB1);
  cudaFree(dC);
  delete[] hA0; delete[] hB0; delete[] hA1; delete[] hB1; delete[] hC;
  return 0;
}
