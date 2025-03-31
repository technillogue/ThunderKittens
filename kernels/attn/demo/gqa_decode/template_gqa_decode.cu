#define KITTENS_TIMINGS

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int QKVO_D = 128, QKVO_D_d2 = QKVO_D/2, QKVO_D_d8 = QKVO_D/8, NUM_ROWS = 32, NUM_ROWS_d2 = NUM_ROWS/2, PAGE_SIZE = 256;
using q_tile           = st_bf<64, QKVO_D>;
using q_global            = kittens::gl<bf16, -1, -1, -1, QKVO_D, q_tile>; // B * R * H * QKVO_D
using kcache_tile         = st_bf<NUM_ROWS, QKVO_D>;
using vcache_tile         = st_bf<NUM_ROWS, QKVO_D>; // we need the v_tile for later
using vcache_tile2        = st_bf<NUM_ROWS, QKVO_D_d2>; // we need the v_tile for later
using kcache_global       = kittens::gl<bf16, 1, -1, PAGE_SIZE, QKVO_D, kcache_tile>; // 1 * #page * pagesize * QKVO_D
using vcache_global       = kittens::gl<bf16, 1, -1, PAGE_SIZE, QKVO_D, vcache_tile>; // 1 * #page * pagesize * QKVO_D
using knew_global         = kittens::gl<bf16, 1, -1, -1, QKVO_D, kcache_tile>;        // 1 * B * lookahead * QKVO_D
using vnew_global         = kittens::gl<bf16, 1, -1, -1, QKVO_D, vcache_tile>;          // 1 * B * lookahead * QKVO_D
using sin_global          = kittens::gl<bf16, 1, 1, -1, QKVO_D_d2>;
using cos_global          = kittens::gl<bf16, 1, 1, -1, QKVO_D_d2>;
using ops_global          = kittens::gl<bf16, 1, -1, -1, 8>;
using instructions_global = kittens::gl<int, 1, -1, -1, 32>;
using table_global        = kittens::gl<int, 1, 1, -1, -1>; // B * (max # pages)
using o_tile              = st_bf<64, QKVO_D>;
using o_tile_fl           = st_fl<16, QKVO_D>;
using o_global            = kittens::gl<bf16, -1, -1, -1, QKVO_D, st_bf<16, QKVO_D_d2>, st_bf<16, QKVO_D_d8>>; // B * NEWTOKENS * H * QKVO_D

template<int Q_HEADS=8>
using o_scratch_global    = kittens::gl<float, -1, -1, Q_HEADS, QKVO_D, st_fl<16, QKVO_D_d8>, st_fl<16,QKVO_D_d2>>; // For partial O's

template<int Q_HEADS=8>
using lvec_scratch_global = kittens::gl<float,  1, -1, -1, Q_HEADS, sv_fl<16>>; // For partial O's
using semaphore_global    = kittens::gl<int,    1,  1,  -1, -1>;            // 1 * 1 * uid * NEWTOKENS

template<int Q_HEADS=8>
struct config {
    struct globals {
        using instructions_global = instructions_global;
        instructions_global instructions;
        q_global Q;
        kcache_global K_cache;
        vcache_global V_cache;
        knew_global K_new;
        vnew_global V_new;
        sin_global sin;
        cos_global cos;
        table_global Table;
        o_global O;
        o_scratch_global<Q_HEADS> O_scratch;
        lvec_scratch_global<Q_HEADS> Lvec_scratch;
        semaphore_global semaphore;
        const float Softmax_scale;
        int tic;
#ifdef KITTENS_TIMINGS
        gl<int, 1, -1, -1, 64> timings;
#endif
        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); } //dim3(Q.batch * ((Q.depth + 3) / 4)); }
        dim3 block() { return dim3((8+4)*WARP_THREADS); }
    };
};

struct location {
    int batch_idx; // batch_idx >=0, otherwise it's the negative index, minus one, into scratch
    int seq_idx;
};
template<int Q_HEADS=8>
struct partial_layout {
    using globals = config<Q_HEADS>::globals;
    struct input_block { kcache_tile kcache; vcache_tile vcache; };
    struct scratch_block { q_tile q; st_bf<64, kcache_tile::rows> att_block; sv_fl<64> max_vec, norm_vec; };
    // always one token per QVO block; if Q_HEADS < 16, then we need to pad
    struct finish_block { st_fl<16, QKVO_D_d2> o[4][2]; sv_fl<16> lvec[4]; };
    struct common_state {
        int uid;
        location dst;
        int q_batch_idx;
        int q_seq_idx;
        int start_pos; // first token handled in this partial
        int end_pos; // One past the last position to load
        int length; // the length of the overall sequence in question (not including new tokens)
    };
    struct consumer_state {
        col_vec<rt_fl<16, kcache_tile::rows>> max_vec, norm_vec;
        rt_fl<16, QKVO_D_d2> o;
    };
};
template<int Q_HEADS=8>
struct partial_template {
    using config = config<Q_HEADS>;
    using layout = partial_layout<Q_HEADS>;
    static constexpr int opcode = 1;
    static constexpr int INPUT_PIPE_STAGES = 3;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
#ifdef KITTENS_TIMINGS
        if(group<12>::laneid() == 0) args.timings[0] = clock64();
#endif
        args.common.uid         =  args.instruction[1];
        args.common.dst         = {args.instruction[2],
                                   args.instruction[3]};
        // which batch is assigned to this worker
        args.common.q_batch_idx =  args.instruction[4];
        // which chunk of new tokens is assigned to this worker (divisible by 4)
        args.common.q_seq_idx   =  args.instruction[5];
        // positions of KV cache assigned to this partial split. Worker will lookup page mapping from Table. positions may not be aligned to PAGE_SIZE.
        args.common.start_pos   =  args.instruction[6];
        args.common.end_pos     =  args.instruction[7];
        // valid seqlen of the assigned batch
        args.common.length      =  args.instruction[8];
        args.num_iters          = cdiv(args.common.end_pos - args.common.start_pos, NUM_ROWS);
        
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            if(args.iter == 1) group<12>::sync(11); // wait for the consumer to finish its setup, before we do the second load.
            if(warpgroup::warpid() == 0) {
                int pos = args.common.start_pos + NUM_ROWS*args.iter;
                int within_page_idx = (pos % PAGE_SIZE) / NUM_ROWS;
                int next_page_id = args.globals.Table[coord<>{args.common.q_batch_idx, pos/PAGE_SIZE}];
                // next page we need to load?
                tma::expect(args.inputs_arrived, args.input.kcache, args.input.vcache);
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.kcache, args.globals.K_cache, {0, next_page_id, within_page_idx, 0}, args.inputs_arrived);
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.vcache, args.globals.V_cache, {0, next_page_id, within_page_idx, 0}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
            }
#ifdef KITTENS_TIMINGS
            else if(warpgroup::laneid() == 32 && args.iter < 24) args.timings[32+args.iter] = clock64();
#endif
            warpgroup::sync(5);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[1] = clock64();
#endif
            // split up Q tile across warps (tokens) and dim (groups) (total 8 ways)
            // each warp loads one token's worth of Q
            auto q_st = subtile_inplace<16, QKVO_D_d2>(args.scratch.q, {warpgroup::warpid(), warpgroup::groupid()});
            auto lookahead_idx = args.common.q_seq_idx + warpgroup::warpid();

            // init local state
            zero(args.state.norm_vec);
            if(args.num_iters > 0) neg_infty(args.state.max_vec);
            else { one(args.state.max_vec); mul(args.state.max_vec, args.state.max_vec, -999999.f); }
            zero(args.state.o);

            // Setup RoPE buffers
            row_vec<rt_bf<16, QKVO_D_d2>> cos_rv;
            row_vec<rt_bf<16, QKVO_D_d2>> sin_rv;
            rt_bf<16, QKVO_D_d2> temp_sin_rt;
            rt_bf<16, QKVO_D_d2> temp_cos_rt;

            if (lookahead_idx < args.globals.K_new.rows()) {
                load_async(q_st, args.globals.Q, {args.common.q_batch_idx, lookahead_idx, 0, warpgroup::groupid()});
                
                load(cos_rv, args.globals.cos, {0, 0, args.common.length + args.common.q_seq_idx + warpgroup::warpid(), 0});
                load(sin_rv, args.globals.sin, {0, 0, args.common.length + args.common.q_seq_idx + warpgroup::warpid(), 0});

                load_async_wait();
                auto other_q_st = subtile_inplace<16, QKVO_D_d2>(args.scratch.q, {warpgroup::warpid(), 1 - warpgroup::groupid()});
                load(temp_cos_rt, q_st);
                load(temp_sin_rt, other_q_st);
    
                // (q1, q2) -> (q1 * cos - q2 * sin, q1 * sin + q2 * cos)
                mul_col(temp_cos_rt, temp_cos_rt, cos_rv);
                mul_col(temp_sin_rt, temp_sin_rt, sin_rv);

                if (warpgroup::groupid() == 0) {
                    sub(temp_cos_rt, temp_cos_rt, temp_sin_rt);
                } else {
                    add(temp_cos_rt, temp_cos_rt, temp_sin_rt);
                }
            }

            group<8>::sync(6);
            store(q_st, temp_cos_rt);
            group<8>::sync(5);

#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[2] = clock64();
#endif
        }
        template<bool do_right_fill, bool do_new_tokens> __device__ static inline void internal_compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            if(args.iter == 0 && args.num_iters > 1) {
                group<12>::arrive(11); // this <12> will allow us to prevent the second producer load from happening before this point.
            }
            const float SOFTMAX_TEMPERATURE = args.globals.Softmax_scale * 1.44269504089f;
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0 && args.iter < 24) args.timings[8+args.iter] = clock64();
#endif

            col_vec<rt_fl<16, kcache_tile::rows>> local_max_vec, local_norm_vec;
            col_vec<rt_fl<16, kcache_tile::rows>> max_vec_last_scaled, max_vec_scaled;

            kittens::barrier<8> cons_barrier(10);

            if(warpgroup::groupid() == 0) {
                // A = Q @ K.T
                // [Q_HEADS, D] @ [NR, D] -> [Q_HEADS, NR] (each warp has its own token of Q, and all share the same KV)
                rt_fl<16, kcache_tile::rows> att_block_fp32;
                warpgroup::mm_ABt(att_block_fp32, args.scratch.q, args.input.kcache);

                copy(local_max_vec,  args.state.max_vec);
                copy(local_norm_vec, args.state.norm_vec);

                mul(max_vec_last_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                warpgroup::mma_async_wait();

                // softmax
                if constexpr (do_right_fill || do_new_tokens) {
                    const int num_valid_tokens = do_right_fill ? 
                        args.common.length - args.common.start_pos - args.iter*NUM_ROWS :
                        args.common.q_seq_idx + warpgroup::warpid() + 1;  // include self for new tokens
                    right_fill(att_block_fp32, att_block_fp32, num_valid_tokens, -9999999999.f);
                }

                row_max(local_max_vec, att_block_fp32, local_max_vec);
                mul(max_vec_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
                sub_row(att_block_fp32, att_block_fp32, max_vec_scaled);
                
                exp2(att_block_fp32, att_block_fp32);
                
                sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                exp2(max_vec_last_scaled, max_vec_last_scaled);
                warpgroup::store(args.scratch.max_vec, max_vec_last_scaled);
                
                mul(local_norm_vec, local_norm_vec, max_vec_last_scaled);
                row_sum(local_norm_vec, att_block_fp32, local_norm_vec);
                warpgroup::store(args.scratch.att_block, att_block_fp32);
                arrive(cons_barrier);
            }
            else {
                arrive_and_wait(cons_barrier);
                warpgroup::load(max_vec_last_scaled, args.scratch.max_vec);
            }

            mul_row(args.state.o, args.state.o, max_vec_last_scaled); // normalize o_reg before mma

            // O += A @ V
            // [Q_HEADS, NUM_ROWS] @ [NUM_ROWS, D] -> [Q_HEADS, D]
            auto (&v_smem)[2] = reinterpret_cast<vcache_tile2(&)[2]>(args.input.vcache);
            warpgroup::mma_AB(args.state.o, args.scratch.att_block, v_smem[warpgroup::groupid()]);

            copy(args.state.max_vec, local_max_vec);
            copy(args.state.norm_vec, local_norm_vec);

            warpgroup::mma_async_wait();
            // if(warpgroup::laneid() == 0) arrive(args.inputs_finished, WARPGROUP_WARPS); // done!
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            if(args.iter >= args.num_iters-2) internal_compute<true, false>(args);
            else internal_compute<false, false>(args);

            // in the last iteration of the task assigned to the rightmost partial of the sequence,
            // we will also handle the new KV tokens, in an extra unscheduled iteration.
            if (args.iter >= args.num_iters-1 && args.common.end_pos == args.common.length) {
                // Q = [NEW_TOKENS * Q_HEADS, D] (we already have this from setup)

                if (warpgroup::groupid() == 0 && warpgroup::warpid() == 0) {
                    // K = [NEW_TOKENS, D] (new KV tokens, load GMEM -> SMEM)
                    // slice [NUM_ROWS, D] from [1, B, R, D]
                    // always load tile at [0, 0], since tile shape works out: NUM_ROWS >= R and NUM_COLS == D
                    load_async(args.input.kcache, args.globals.K_new, {0, args.common.q_batch_idx, 0, 0});
                    load_async(args.input.vcache, args.globals.V_new, {0, args.common.q_batch_idx, 0, 0});
                    load_async_wait();

                    auto kcache_st_0 = subtile_inplace<NUM_ROWS_d2, QKVO_D_d2>(args.input.kcache, {0, 0});
                    auto kcache_st_1 = subtile_inplace<NUM_ROWS_d2, QKVO_D_d2>(args.input.kcache, {0, 1});

                    rt_bf<NUM_ROWS_d2, QKVO_D_d2> k_rt_0;
                    rt_bf<NUM_ROWS_d2, QKVO_D_d2> k_rt_1;
                    rt_bf<NUM_ROWS_d2, QKVO_D_d2> cos_rt;
                    rt_bf<NUM_ROWS_d2, QKVO_D_d2> sin_rt;
                    rt_bf<NUM_ROWS_d2, QKVO_D_d2> k_rt_0_sin;
                    rt_bf<NUM_ROWS_d2, QKVO_D_d2> k_rt_0_cos;
                    rt_bf<NUM_ROWS_d2, QKVO_D_d2> k_rt_1_sin;
                    rt_bf<NUM_ROWS_d2, QKVO_D_d2> k_rt_1_cos;

                    load(k_rt_0, kcache_st_0);
                    load(k_rt_1, kcache_st_1);
                    load(cos_rt, args.globals.cos, coord<>{0, 0, args.common.length + args.common.q_seq_idx, 0});
                    load(sin_rt, args.globals.sin, coord<>{0, 0, args.common.length + args.common.q_seq_idx, 0});

                    mul(k_rt_0_cos, k_rt_0, cos_rt); // k_rt_0_cos = k_rt_0 * cos
                    mul(k_rt_0_sin, k_rt_0, sin_rt); // k_rt_0_sin = k_rt_0 * sin
                    mul(k_rt_1_cos, k_rt_1, cos_rt); // k_rt_1_cos = k_rt_1 * cos
                    mul(k_rt_1_sin, k_rt_1, sin_rt); // k_rt_1_sin = k_rt_1 * sin

                    // (k_rt_0, k_rt_1) -> (k_rt_0 * cos - k_rt_1 * sin, k_rt_0 * sin + k_rt_1 * cos)
                    sub(k_rt_0, k_rt_0_cos, k_rt_1_sin); // k_rt_0 = k_rt_0_cos - k_rt_1_sin = k_rt_0 * cos - k_rt_1 * sin
                    add(k_rt_1, k_rt_1_cos, k_rt_0_sin); // k_rt_1 = k_rt_1_cos + k_rt_0_sin = k_rt_1 * cos + k_rt_0 * sin
                    
                    store(kcache_st_0, k_rt_0);
                    store(kcache_st_1, k_rt_1);
                }
                group<8>::sync(17);

                // QK -> [NEW_TOKENS * Q_HEADS, NEW_TOKENS]
                // V = [NEW_TOKENS, D]
                // QK @ V -> [NEW_TOKENS * Q_HEADS, D]
                internal_compute<false, true>(args);

                if (warpgroup::groupid() == 0 and warpgroup::warpid() == 0) {
                    // write out KV update

                    auto num_new_tokens = args.globals.K_new.rows();
                    auto eos_page_idx = args.common.length/PAGE_SIZE;
                    auto space_left_in_page = PAGE_SIZE - (args.common.length % PAGE_SIZE);

                    // will always write at least 1 token to existing page
                    auto tokens_in_trailing_page = min(num_new_tokens, space_left_in_page);
                    auto eos_page_addr = args.globals.Table[coord<>{args.common.q_batch_idx, eos_page_idx}];
                    auto row_offset_within_page = args.common.length % PAGE_SIZE;
                    kittens::store_masked(args.globals.K_cache, args.input.kcache, {0, eos_page_addr, 0, 0}, row_offset_within_page, 0, tokens_in_trailing_page);
                    kittens::store_masked(args.globals.V_cache, args.input.vcache, {0, eos_page_addr, 0, 0}, row_offset_within_page, 0, tokens_in_trailing_page);
                    
                    // write remaining tokens to next page
                    auto tokens_in_next_page = num_new_tokens - tokens_in_trailing_page;
                    if (tokens_in_next_page > 0) {
                        auto next_page_addr = args.globals.Table[coord<>{args.common.q_batch_idx, eos_page_idx + 1}];
                        kittens::store_masked(args.globals.K_cache, args.input.kcache, {0, next_page_addr, 0, 0}, 0, tokens_in_trailing_page, tokens_in_next_page);
                        kittens::store_masked(args.globals.V_cache, args.input.vcache, {0, next_page_addr, 0, 0}, 0, tokens_in_trailing_page, tokens_in_next_page);
                    }
                }
                group<8>::sync(17);
            }

            if(warpgroup::laneid() == 0) arrive(args.inputs_finished, WARPGROUP_WARPS); // done!

        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            col_vec<rt_fl<16, kcache_tile::rows>> local_max_vec, local_norm_vec;

            copy(local_norm_vec, args.state.norm_vec);
            copy(local_max_vec, args.state.max_vec);

#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[62] = clock64(); // Start of store out.
#endif

            if (warpgroup::groupid() == 0) warpgroup::store(args.scratch.norm_vec, local_norm_vec);
            group<8>::sync(10);
            if(warpgroup::groupid() == 1) warpgroup::load(local_norm_vec, args.scratch.norm_vec);
            div_row(args.state.o, args.state.o, local_norm_vec);

            if(args.common.dst.batch_idx >= 0) { // batch is meaningful
                auto &o_smem = reinterpret_cast<st_bf<16, QKVO_D_d2>&>(args.finish.o[warpgroup::warpid()][warpgroup::groupid()]);
                store(o_smem, args.state.o);
                __syncwarp();
                tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx+warpgroup::warpid(), 0, warpgroup::groupid()});
            }
            else { // write out directly to O scratch, without going through smem
                if(warpgroup::groupid() == 0) {
                    mul(local_max_vec, local_max_vec, args.globals.Softmax_scale * 1.44269504089f);
                    log2(local_norm_vec, local_norm_vec);
                    add(local_norm_vec, local_norm_vec, local_max_vec); // l_vec = log2(norm_vec) + max_vec
                    store(args.finish.lvec[warpgroup::warpid()], local_norm_vec);
                    __syncwarp();
                    tma::store_async<cache_policy::EVICT_LAST>(args.globals.Lvec_scratch, args.finish.lvec[warpgroup::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpgroup::warpid(), 0});
                }
                store(args.finish.o[warpgroup::warpid()][warpgroup::groupid()], args.state.o);
                __syncwarp();
                tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(args.globals.O_scratch, args.finish.o[warpgroup::warpid()][warpgroup::groupid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpgroup::warpid(), 0, warpgroup::groupid()});
            }
            if(group<8>::warpid() == 0) tma::store_async_wait(); // not just read wait
            asm volatile("fence.sc.cta;\n"); // Can't reorder across this boundary
            group<8>::sync(10);
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() < 4 && args.common.dst.seq_idx + group<8>::laneid() < args.globals.O_scratch.depth()) {
                    // Todo: this can probably replaced by a st.async, which may prevent an expensive wait on the final finish barrier.
                    args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx + group<8>::laneid()}] = args.globals.tic;
                    // For blackwell
                    // asm volatile(
                    //     "st.async.global.b32 [%0], %1;\n"
                    // ::  "l"(&args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx + group<8>::laneid()}]), "r"(args.globals.tic)
                    // :   "memory"
                    // );
                }
            }
            if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
#ifdef KITTENS_TIMINGS
            else if(group<8>::laneid() == 32) args.timings[63] = clock64();
#endif
        }
    };
};

template<int Q_HEADS=8>
struct reduction_layout {
    using globals = config<Q_HEADS>::globals;
    struct input_block   { st_fl<16, QKVO_D_d8> o[8]; sv_fl<16> lvec; sv_fl<16> padding[15]; };
    struct scratch_block { st_fl<16, QKVO_D_d8> o[8]; sv_fl<16> lvec; semaphore producer_block; }; // used both for setup load and finish store
    struct common_state {
        int uid;
        // int num_iters; // same as the number of active load_uid's, marked here for instruction clarity but we just use args.num_iters instead.
        location dst; // again, negative batch means we're writing to O scratch, seq_idx is consistent
        int src_uid;
    };
    struct consumer_state {
        rt_fl<16, QKVO_D_d8> o;
        col_vec<rt_fl<16, kcache_tile::rows>> lvec;
    };
};

template<int Q_HEADS=8>
struct reduction_template {
    using config = config<Q_HEADS>;
    using layout = reduction_layout<Q_HEADS>;
    static constexpr int opcode = 2;
    static constexpr int INPUT_PIPE_STAGES = 4;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
#ifdef KITTENS_TIMINGS
        if(group<12>::laneid() == 0) args.timings[0] = clock64();
#endif
        args.common.uid     =  args.instruction[1];
        args.num_iters      =  args.instruction[2];
        args.common.dst     = {args.instruction[3],
                               args.instruction[4]};
        args.common.src_uid =  args.instruction[5];
        group<12>::sync(7);
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            // if(args.iter == 1) group<12>::sync(8);
            if(warpgroup::warpid() == args.iter%4) {
                // spinloop until we're ready
                int load_uid = args.instruction[6+args.iter];
                if(laneid() == 0) while(*(volatile int*)&args.globals.semaphore[{load_uid, args.common.dst.seq_idx}] != args.globals.tic) {}
                __syncwarp();
#ifdef KITTENS_TIMINGS
                if(laneid() == 0 && args.iter < 24) args.timings[32+args.iter] = clock64();
#endif
                // next page we need to load?
                tma::expect(args.inputs_arrived, args.input.o, args.input.lvec);
                #pragma unroll
                for(int i = 0; i < 8; i++) {
                    tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.o[i], args.globals.O_scratch, {load_uid, args.common.dst.seq_idx, 0, i}, args.inputs_arrived);
                }
                tma::load_async<cache_policy::EVICT_FIRST>(args.input.lvec, args.globals.Lvec_scratch, {load_uid, args.common.dst.seq_idx, 0}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
            }
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            // If we are doing a reduction, we need to spinloop until we have confirmation that all the partial results have been written out.
            if(threadIdx.x == 0) { // easier to have a single thread spin
                while(*(volatile int*)&args.globals.semaphore[{args.common.src_uid, args.common.dst.seq_idx}] != args.globals.tic) {} // note volatile, L1 is not guaranteed to be coherent.
            }
            group<8>::sync(11); // all warps must sync here.
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[1] = clock64();
#endif
            load_async(args.scratch.o[group<8>::warpid()], args.globals.O_scratch, {args.common.src_uid, args.common.dst.seq_idx, 0, group<8>::warpid()});
            if(warpid() == 0) {
                load_async(args.scratch.lvec, args.globals.Lvec_scratch, {args.common.src_uid, args.common.dst.seq_idx, 0});
            }
            load_async_wait();
            __syncwarp();
            load(args.state.o, args.scratch.o[group<8>::warpid()]);
            group<8>::sync(11); // we use this to also stall the producer until the consumer is ready.
            // group<12>::sync(9);
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[2] = clock64();
#endif
            load(args.state.lvec, args.scratch.lvec);
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0 && args.iter < 24) args.timings[8+args.iter] = clock64();
#endif
            col_vec<rt_fl<16, kcache_tile::rows>> lvec, max_lvec, sum_lvec;
            rt_fl<16, QKVO_D_d8> o;
            load(o, args.input.o[group<8>::warpid()]);
            load(lvec, args.input.lvec);
            __syncwarp();
            if(laneid() == 0) arrive(args.inputs_finished); // done!
            max(max_lvec, args.state.lvec, lvec);
            sub(args.state.lvec, args.state.lvec, max_lvec);
            sub(lvec, lvec, max_lvec);
            exp2(args.state.lvec, args.state.lvec);
            exp2(lvec, lvec);
            add(sum_lvec, args.state.lvec, lvec);
            div(args.state.lvec, args.state.lvec, sum_lvec);
            div(lvec, lvec, sum_lvec);
            mul_row(args.state.o, args.state.o, args.state.lvec);
            mul_row(o, o, lvec);
            add(args.state.o, args.state.o, o);
            log2(sum_lvec, sum_lvec);
            add(args.state.lvec, sum_lvec, max_lvec);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[62] = clock64();
#endif
            if(args.common.dst.batch_idx >= 0) {
                auto &o_smem = reinterpret_cast<st_bf<16, QKVO_D_d8>&>(args.scratch.o[group<8>::warpid()]);
                store(o_smem, args.state.o);
                __syncwarp();
                tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx, 0, group<8>::warpid()});
            }
            else {
                store(args.scratch.o[group<8>::warpid()], args.state.o);
                if(group<8>::warpid() == 0) store(args.scratch.lvec, args.state.lvec);
                __syncwarp();
                tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(args.globals.O_scratch, args.scratch.o[group<8>::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, 0, group<8>::warpid()});
                if(group<8>::warpid() == 0) tma::store_async<cache_policy::EVICT_LAST>(args.globals.Lvec_scratch, args.scratch.lvec, {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, 0});
            }
            tma::store_async_wait();
            group<8>::sync(11);
            // Increment the semaphore for the next stage, if this is not the last one.
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() == 0) {
                    args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx}] = args.globals.tic;
                }
            }
            if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
#ifdef KITTENS_TIMINGS
            else if(group<8>::laneid() == 32) args.timings[63] = clock64();
#endif
        }
    };
};

#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Timing constants (in microseconds)
constexpr float PARTIAL_STARTUP_TIME = 2.5f;         // Startup time for partial operations
constexpr float PARTIAL_WRITEOUT_TIME = 1.2f;        // Writeout time for partial operations
constexpr float PARTIAL_COST_PER_STEP = 0.6f;       // Cost per step (per 32 tokens) for partial operations
constexpr float PARTIAL_OVERHEAD = PARTIAL_STARTUP_TIME + PARTIAL_WRITEOUT_TIME; // Total overhead for a partial operation.

constexpr float REDUCTION_STARTUP_TIME = 1.0f;       // Startup time for reduction operations
// constexpr float REDUCTION_WRITEOUT_TIME = 1.0f;   // Writeout time for reduction operations, not used so commented to disable warnings.
constexpr float REDUCTION_PRODUCER_LATENCY = 1.0f;   // Latency between a producer load and when the consumer can access it.
constexpr float REDUCTION_COST_PER_STEP = 0.3f;      // Cost per reduction step

// constexpr float SYNCHRONIZATION_COST = 0.5f;      // Synchronization cost between dependent operations, not used so commented to disable warnings.

float get_quality(const std::vector<float>& next_times_input, int num_processors, int num_tokens, int seq_length) {
    int num_partial_steps = cdiv(seq_length, 32);

    if (next_times_input.size() > num_processors) {
        // This particular scheduler is just not set up to deal with these situations.
        return -999999999.0f;
    }
    
    // Copy the input times so we can modify them
    std::vector<float> next_times = next_times_input;
    std::sort(next_times.begin(), next_times.end(), std::greater<float>());

    std::vector<float> partial_times;
    for (int i = 0; i < num_processors; i++) {
        next_times[i%next_times.size()] -= REDUCTION_COST_PER_STEP;
        partial_times.push_back(next_times[i%next_times.size()]);
    }

    // We also have to account for the fact that some of these partials are going to be forced to start earlier than the coscheduled reduction.
    // The number of these is equal to the number of reduction ops * the number of tokens, since those are each handled independently.
    std::sort(partial_times.begin(), partial_times.end()); // Thankfully we can pick the worst ones to be forced to start earlier.
    
    for (size_t j = 0; j < next_times.size(); j++) {
        float actual_start_time = next_times[j] + REDUCTION_PRODUCER_LATENCY - REDUCTION_STARTUP_TIME; // When does this instruction actually start?
        for (int k = 0; k < num_tokens; k++) {
            if (num_tokens * j + k < partial_times.size()) {
                partial_times[num_tokens * j + k] = actual_start_time;
            }
        }
    }
    
    // Now that we know when the partials are going to start, we can start to assign the steps of the work.
    std::sort(partial_times.begin(), partial_times.end(), std::greater<float>()); // Largest to smallest.
    
    float min_value = partial_times.back();
    for(int i = 0; i < partial_times.size(); i++) {
        if(num_partial_steps > 0) {
            int num_steps_alloc = std::min(num_partial_steps, (int)(round((partial_times[i]-min_value) / PARTIAL_COST_PER_STEP)));
            num_partial_steps -= num_steps_alloc;
            partial_times[i] -= num_steps_alloc * PARTIAL_COST_PER_STEP;
            if(num_steps_alloc > 0) partial_times[i] -= PARTIAL_OVERHEAD;
        }
    }

    int full_passes = num_partial_steps / partial_times.size();
    int remainder = num_partial_steps - (full_passes * partial_times.size());

    std::sort(partial_times.begin(), partial_times.end(), std::greater<float>());
    min_value = 9999999999.0f;
    for(int i = 0; i < remainder; i++){
        float f = partial_times[i] - PARTIAL_COST_PER_STEP * (full_passes+1);
        if(f < min_value) min_value = f;
    }
    for(int i = remainder; i < partial_times.size(); i++) {
        float f = partial_times[i] - PARTIAL_COST_PER_STEP * full_passes;
        if(f < min_value) min_value = f;
    }

    return min_value;
}

PYBIND11_MODULE(gqa_decode, m) {
    m.doc() = "gqa_decode python module";
    kittens::py::bind_kernel<interpreter::kernel<config<8>, partial_template<8>, reduction_template<8>>>(m, "gqa_decode_8_heads",
        &config<8>::globals::instructions,
        &config<8>::globals::Q,
        &config<8>::globals::K_cache,
        &config<8>::globals::V_cache,
        &config<8>::globals::K_new,
        &config<8>::globals::V_new,
        &config<8>::globals::sin,
        &config<8>::globals::cos,
        &config<8>::globals::Table,
        &config<8>::globals::O,
        &config<8>::globals::O_scratch,
        &config<8>::globals::Lvec_scratch,
        &config<8>::globals::semaphore,
        &config<8>::globals::Softmax_scale,
        &config<8>::globals::tic
#ifdef KITTENS_TIMINGS
        , &config<8>::globals::timings
#endif
    );
    m.def("__get_quality__", &get_quality, 
        "An internal utility function for generating efficient schedules.",
        pybind11::arg("next_times"), 
        pybind11::arg("num_processors"), 
        pybind11::arg("num_tokens"), 
        pybind11::arg("seq_length")
    );
}