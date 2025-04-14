#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define _row 32
#define _col 32

struct micro_globals {
    using _gl  = gl<float, -1, -1, -1, 64, st_fl<_row, _col>>;
    _gl x;
    kittens::optional<_gl> y;
    _gl o;

    dim3 grid()  { return dim3(x.batch(), x.depth(), x.rows()); }
    dim3 block() { return dim3(x.cols()); }
    int dynamic_shared_memory() {return 50480;}
};

__global__ //__launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();
    st_fl<_row, _col> (&y_s) = al.allocate<st_fl<_row, _col>>();
    st_fl<_row, _col> (&o_s) = al.allocate<st_fl<_row, _col>>();

    // register memory 
    rt_fl<_row, _col> x_reg_fl;
    rt_fl<_row, _col> y_reg_fl;

    // load from HBM to shared
    load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg_fl, x_s);
    __syncthreads();

    if (g.y.has_value()) {
        // Now load Y.
        // load from HBM to shared
        //load(y_s, g.y, {0,0,0,0});
        load(y_s, g.y.value(), {0, 0, 0, 0});
        __syncthreads();

        // load from shared to register
        load(y_reg_fl, y_s);
        __syncthreads();

        // x (dst) = x (src b) + x (src a)
        add(x_reg_fl, x_reg_fl, y_reg_fl);
        __syncthreads();
    } 

    // store from register to shared
    store(o_s, x_reg_fl);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {0, 0, 0, 0});
    __syncthreads();
}

PYBIND11_MODULE(micro_add, m) {
    m.doc() = "micro_add python module";
    kittens::py::bind_kernel<micro_tk>(m, "add_mats", &micro_globals::x, &micro_globals::y, &micro_globals::o);
    kittens::py::bind_kernel_named<micro_tk>(m, "add_mats", &micro_globals::x, py_arg("y", &micro_globals::y), py_arg("o", &micro_globals::o));
}
