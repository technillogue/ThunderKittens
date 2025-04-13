#pragma once

#include "util.cuh"
#include <pybind11/pybind11.h>
#include <type_traits> // Needed for remove_cvref_t

// Helper macro to create a unique static constexpr char array pointer for a string literal
// not sure if this is *really* needed
#define KITTENS_PY_ARG_NAME(s) \
    [] { static constexpr char value[] = s; return value; }()


// The wrapper struct template (stores value T and name pointer N)
template<typename T, const char* N>
struct kittens_py_named { // Using a more specific name to avoid potential conflicts
    T value;
    static constexpr const char* name_str = N;

    // Constructor for easy initialization from the underlying type
    kittens_py_named(T val = {}) : value(val) {}

    // Allow implicit conversion back to the underlying type
    operator T&() { return value; }
    operator const T&() const { return value; }

    // Explicit accessors if needed
    T& get() { return value; }
    const T& get() const { return value; }
};

// Convenience macro for declaring named members
#define NAMED(TheType, TheNameStringLiteral) \
    kittens_py_named<TheType, KITTENS_PY_ARG_NAME(TheNameStringLiteral)>


namespace kittens {
namespace py {

// --- NEW: Traits and Helpers for Named Arguments ---
namespace detail {

// --- Name Trait ---
// Base case: No name
template<typename T>
struct name_trait {
    static constexpr const char* name = nullptr;
};

// Specialization for our named wrapper
template<typename T, const char* N>
struct name_trait<kittens_py_named<T, N>> {
    static constexpr const char* name = N;
};

// Helper to get name or "" for py::arg
template<typename MemberType>
constexpr const char* get_py_arg_name() {
    // Use remove_cvref to handle potential const/volatile/reference qualifiers
    constexpr const char* extracted_name = name_trait<std::remove_cvref_t<MemberType>>::name;
    if constexpr (extracted_name != nullptr) {
        return extracted_name;
    } else {
        // Return an empty string if no name is defined.
        // py::arg("") implies positional-or-keyword with no specific name hint.
        return "";
    }
}

// --- Underlying Type Trait ---
// Base case: Type is itself
template<typename T>
struct underlying_type_trait {
    using type = T;
};

// Specialization for our named wrapper
template<typename T, const char* N>
struct underlying_type_trait<kittens_py_named<T, N>> {
    using type = T; // Extract the original type T
};

template<typename T>
using underlying_type_t = typename underlying_type_trait<std::remove_cvref_t<T>>::type;

} // namespace detail

template<typename T> struct from_object {
    static T make(pybind11::object obj) {
        return obj.cast<T>();
    }
};

template<typename T, const char* N>
struct from_object<kittens_py_named<T, N>> {
    static kittens_py_named<T, N> make(pybind11::object obj) {
        // Delegate to from_object for the underlying type T
        // Note: We use `detail::underlying_type_t` here just in case T itself
        // could be complex, but usually just `T` is fine. Using `T` directly is simpler.
        // T underlying_value = from_object<detail::underlying_type_t<T>>::make(obj);
        T underlying_value = from_object<T>::make(obj); // Simpler version

        // Wrap the result in the 'kittens_py_named' struct
        return kittens_py_named<T, N>{underlying_value};
    }
};

template<ducks::gl::all GL> struct from_object<kittens::optional<GL>> {
    static kittens::optional<GL> make(pybind11::object obj) {
        printf("inside optional from_object, obj is none? %d\n", obj.is_none());
        if (obj.is_none()) {
            return kittens::optional<GL>(std::nullopt);
        }
        return kittens::optional<GL>(from_object<GL>::make(obj));
    }
};

template<ducks::gl::all GL> struct from_object<GL> {
    static GL make(pybind11::object obj) {
        // Check if argument is a torch.Tensor
        if (pybind11::hasattr(obj, "__class__") && 
            obj.attr("__class__").attr("__name__").cast<std::string>() == "Tensor") {
        
            // Check if tensor is contiguous
            if (!obj.attr("is_contiguous")().cast<bool>()) {
                throw std::runtime_error("Tensor must be contiguous");
            }
            if (obj.attr("device").attr("type").cast<std::string>() == "cpu") {
                throw std::runtime_error("Tensor must be on CUDA device");
            }
            
            // Get shape, pad with 1s if needed
            std::array<int, 4> shape = {1, 1, 1, 1};
            auto py_shape = obj.attr("shape").cast<pybind11::tuple>();
            size_t dims = py_shape.size();
            if (dims > 4) {
                throw std::runtime_error("Expected Tensor.ndim <= 4");
            }
            for (size_t i = 0; i < dims; ++i) {
                shape[4 - dims + i] = pybind11::cast<int>(py_shape[i]);
            }
            
            // Get data pointer using data_ptr()
            uint64_t data_ptr = obj.attr("data_ptr")().cast<uint64_t>();
            
            // Create GL object using make_gl
            return make_gl<GL>(data_ptr, shape[0], shape[1], shape[2], shape[3]);
        }
        throw std::runtime_error("Expected a torch.Tensor");
    }
};

template<typename T> concept has_dynamic_shared_memory = requires(T t) { { t.dynamic_shared_memory() } -> std::convertible_to<int>; };

template<typename> struct trait;
template<typename MT, typename T> struct trait<MT T::*> { using member_type = MT; using type = T; };
template<typename> using object = pybind11::object;
template<auto kernel, typename TGlobal> static void bind_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {

    m.def(name, [](object<decltype(member_ptrs)>... args) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__>>>(__g__);
        } else {
            kernel<<<__g__.grid(), __g__.block()>>>(__g__);
        }
    });
}
template<auto function, typename TGlobal> static void bind_function(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        function(__g__);
    });
}

// new
template<auto kernel, typename TGlobal>
static void bind_kernel_named(pybind11::module_ m, const char* func_name, auto TGlobal::*... member_ptrs) {

    auto kernel_lambda = [](object<decltype(member_ptrs)>... args) {
        // Construct TGlobal using from_object with the *actual* member type (T or named<T,N>)
        TGlobal __g__ {
            from_object<
                typename trait<decltype(member_ptrs)>::member_type
            >::make(args)...
        };

        // Launch the kernel
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            cudaError_t err_attr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
             if (err_attr != cudaSuccess) {
                 throw std::runtime_error(std::string("Failed to set dynamic shared memory: ") + cudaGetErrorString(err_attr));
             }
            kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__>>>(__g__);
        } else {
            kernel<<<__g__.grid(), __g__.block()>>>(__g__);
        }
        CHECK_CUDA_ERROR(cudaGetLastError())
    };

    // Define the function using m.def, unpacking the py::arg names
    m.def(func_name,
          kernel_lambda,
          // Generate py::arg("name") or py::arg("") for each member
          pybind11::arg(
              detail::get_py_arg_name< // Use helper to get name string...
                  typename trait<decltype(member_ptrs)>::member_type // ...from the actual member type (T or named<T,N>)
              >()
          )... // ...for each member pointer (pack expansion)
    );
}

} // namespace py
} // namespace kittens
