#pragma once

#include "util.cuh"
#include <pybind11/pybind11.h>
#include <type_traits> // Needed for remove_cvref_t
#include <utility> 

namespace kittens {
namespace py {

template<typename> struct trait;
template<typename MT, typename T> struct trait<MT T::*> { using member_type = MT; using type = T; };
template<typename> using object = pybind11::object;

// Base struct to hold any member pointer type
template <typename TGlobal, typename MemberType>
struct KernelArgBase {
    MemberType TGlobal::*ptr;
    constexpr explicit KernelArgBase(MemberType TGlobal::*p) : ptr(p) {}
};

// Struct for an unnamed argument (just holds the pointer)
// Note: This struct isn't strictly *required* anymore with the raw pointer handling,
// but py_arg(&ptr) overload returning this can provide symmetry.
template <typename TGlobal, typename MemberType>
struct UnnamedKernelArg : KernelArgBase<TGlobal, MemberType> {
    static constexpr const char* name_str = ""; // Empty name for py::arg("")
    constexpr explicit UnnamedKernelArg(MemberType TGlobal::*p) : KernelArgBase<TGlobal, MemberType>(p) {}
};

// Struct for a named argument (holds pointer and name)
template <typename TGlobal, typename MemberType>
struct NamedKernelArg : KernelArgBase<TGlobal, MemberType> {
    const char* name_str; // Name is runtime string literal passed via py_arg
    constexpr NamedKernelArg(const char* n, MemberType TGlobal::*p) : KernelArgBase<TGlobal, MemberType>(p), name_str(n) {}
};

// Helper function to create a named argument wrapper (like py::arg)
template <typename TGlobal, typename MemberType>
constexpr NamedKernelArg<TGlobal, MemberType> py_arg(const char* name, MemberType TGlobal::*ptr) {
    return NamedKernelArg<TGlobal, MemberType>(name, ptr);
}

// Helper function/overload for unnamed arguments (returns the UnnamedKernelArg wrapper)
template <typename TGlobal, typename MemberType>
constexpr UnnamedKernelArg<TGlobal, MemberType> py_arg(MemberType TGlobal::*ptr) {
    return UnnamedKernelArg<TGlobal, MemberType>(ptr);
}
// not sure if needed
// Concept to check if a type is one of our *argument wrapper* structs (specifically NamedKernelArg or UnnamedKernelArg)
template<typename T>
concept is_kernel_arg_wrapper = requires {
    // Check if it inherits from KernelArgBase (or has the members directly)
    typename T::pointer_type; // Example: Add a dummy type alias to the wrappers if needed
} && (std::is_base_of_v<KernelArgBase<typename T::global_type, typename T::member_type>, T>); // Adjust based on how wrappers are defined

// More robust check using type traits or specific markers might be needed depending on final struct definitions
template<typename T>
concept is_kernel_arg = std::is_base_of_v<KernelArgBase<typename T::global_type, typename T::member_type>, T>; // Simplified check if base works
// If using the simple structs above:
template<typename T> concept is_kernel_arg_wrapper_struct = requires (T t) {
    { t.ptr } -> std::convertible_to<void*>; // Check if it has a 'ptr' member
    { t.name_str }; // Check if it has 'name_str' (static or member)
};


// --- Traits and Helpers within detail namespace ---
namespace detail {
    // Concept to check if a type is already one of our wrappers
    template<typename T>
    concept is_kernel_arg_wrapper = requires(T t) {
        typename T::global_type;
        typename T::member_type;
        { t.ptr } -> std::convertible_to<void*>;
    } && (std::is_same_v<T, NamedKernelArg<typename T::global_type, typename T::member_type>> ||
          std::is_same_v<T, UnnamedKernelArg<typename T::global_type, typename T::member_type>>);

    // Helper to wrap raw pointers into UnnamedKernelArg
    template <typename ArgType>
    constexpr auto wrap_if_needed(const ArgType& arg) {
        // Check if it's already one of our wrapper structs
        if constexpr (is_kernel_arg_wrapper<std::decay_t<ArgType>>) {
            return arg; // Pass through if already wrapped
        } else {
            // Otherwise, assume it's a raw member pointer and wrap it
            using ArgDecayed = std::decay_t<ArgType>;
            static_assert(std::is_member_object_pointer_v<ArgDecayed>,
                          "Argument must be a member object pointer or result of py_arg()");

            // Extract TGlobal and MemberType using the trait struct
            using PtrTraits = kittens::py::trait<ArgDecayed>;
            using TGlobal = typename PtrTraits::type;
            using MemberType = typename PtrTraits::member_type;

            // Wrap the raw pointer in UnnamedKernelArg
            return UnnamedKernelArg<TGlobal, MemberType>(arg);
        }
    }

    // Helper to get the name string ("" for raw pointers)
    template <typename ArgType>
    constexpr const char* get_bind_arg_name(const ArgType& arg) {
        // Check if it's one of our wrapper structs (NamedKernelArg or UnnamedKernelArg)
        if constexpr (is_kernel_arg_wrapper_struct<ArgType>) {
            return arg.name_str; // Return its stored name (could be "" for Unnamed)
        } else {
            // Otherwise, assume it's a raw member pointer and return "" for unnamed
            static_assert(std::is_member_object_pointer_v<std::decay_t<ArgType>>, // Use decay_t
                          "Argument must be a member object pointer or result of py_arg()");
            return ""; // Treat raw pointers as unnamed
        }
    }

    // Helper to get the underlying member pointer itself
    template <typename ArgType>
    constexpr auto get_bind_arg_pointer(const ArgType& arg) {
         // Check if it's one of our wrapper structs
         if constexpr (is_kernel_arg_wrapper_struct<ArgType>) {
            return arg.ptr; // Return the stored pointer
        } else {
            // Otherwise, it's already the raw member pointer
            static_assert(std::is_member_object_pointer_v<std::decay_t<ArgType>>, // Use decay_t
                          "Argument must be a member object pointer or result of py_arg()");
            return arg; // Return the raw pointer itself
        }
    }


    // --- Implementation function that takes ONLY wrappers ---
    // (This contains the lambda and m.def logic from the previous "force wrapping" attempt)
    template <auto kernel, typename TGlobal, typename... WrappedArgs>
    static void bind_kernel_named_impl(pybind11::module_ m, const char* func_name, WrappedArgs... wrapped_args)
    {
        // Static assert to double-check (should be guaranteed by caller)
        static_assert((detail::is_kernel_arg_wrapper<WrappedArgs> && ...),
                      "Internal error: bind_kernel_named_impl called with non-wrapper arguments");

        // The simple lambda captures the homogeneous pack of wrappers
        auto kernel_lambda = [=](pybind11::object... py_objs) {
            auto py_args_tuple = std::forward_as_tuple(py_objs...);
            constexpr size_t num_py_args = std::tuple_size_v<decltype(py_args_tuple)>;
            constexpr size_t num_expected_args = sizeof...(WrappedArgs);

            // --- Use tuple size for arity check ---
            if (num_expected_args != num_py_args) {
                 throw pybind11::type_error("Kernel expected " + std::to_string(num_expected_args)
                                            + " arguments, but got " + std::to_string(num_py_args));
            }

            // --- Workaround: Construct TGlobal using index_sequence ---
            // Helper lambda defined inside to capture necessary variables
            auto construct_globals = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                // Create tuple of the captured wrappers to access them by index
                auto captured_wrappers_tuple = std::make_tuple(wrapped_args...);
                return TGlobal{
                    from_object<
                        // Get pointer via impl helper, get type via trait (qualified)
                        typename kittens::py::trait<decltype(detail::get_bind_arg_pointer_impl(std::get<Is>(captured_wrappers_tuple)))>::member_type
                    >::make(std::get<Is>(py_args_tuple))... // Expand using py_args_tuple elements
                };
            };
            // Call the helper to initialize globals
            TGlobal __g__ = construct_globals(std::make_index_sequence<num_expected_args>{});

            // Kernel launch logic (same as before, with error checking)
            if constexpr (has_dynamic_shared_memory<TGlobal>) {
                int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
                cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
                CHECK_CUDA_ERROR(err);
                kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__>>>(__g__);
            } else {
                kernel<<<__g__.grid(), __g__.block()>>>(__g__);
            }
            CHECK_CUDA_ERROR(cudaGetLastError());
        };

        // m.def uses the OVERLOADED helper on the original wrapped_args pack
        m.def(func_name,
              kernel_lambda,
              pybind11::arg(detail::get_bind_arg_name_impl(wrapped_args))...
        );
    }
} // namespace detail


template<typename T> struct from_object {
    static T make(pybind11::object obj) {
        return obj.cast<T>();
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

template<auto kernel, typename TGlobal>
static void bind_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {

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

// --- Kernel Binding Functions ---

// --- Public bind_kernel_named: Accepts mixed args, wraps, then calls impl ---
template <auto kernel, typename TGlobal, typename... Args>
static void bind_kernel_named(pybind11::module_ m, const char* func_name, Args... args)
{
    // Call the implementation function, applying wrap_if_needed to each argument.
    // This creates a new, homogeneous pack expansion of wrappers.
    detail::bind_kernel_named_impl<kernel, TGlobal>(
        m,
        func_name,
        detail::wrap_if_needed(args)... // Apply wrapping transformation HERE
    );
}

} // namespace py
} // namespace kittens
