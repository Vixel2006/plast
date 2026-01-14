#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
extern "C" {
#include "arena.h"
#include "arena_cuda.h"
#include "tensor.h"
#include "node.h"
#include "graph.h"
#include "op.h"
#include "optimizers/sgd.h"
#include "optimizers/zero_grad.h"
}

namespace py = pybind11;

// Wrapper for DTYPE
enum PyDType {
    INT32_PY = INT32,
    FLOAT32_PY = FLOAT32
};

PYBIND11_MODULE(plast_core, m) {
    m.doc() = "Plast core library bindings";

    py::enum_<DEVICE>(m, "Device")
        .value("CPU", CPU)
        .value("CUDA", CUDA)
        .export_values();

    py::enum_<DTYPE>(m, "DType")
        .value("Int32", INT32)
        .value("Float32", FLOAT32)
        .export_values();

    py::class_<Arena>(m, "Arena")
        .def(py::init([](u64 capacity, DEVICE device) {
            return arena_create(capacity, device);
        }))
        .def("release", &arena_release)
        .def("reset", &arena_reset);

    py::class_<Tensor>(m, "Tensor")
        .def_property_readonly("ndim", [](const Tensor &t) { return t.ndim; })
        .def_property_readonly("shape", [](const Tensor &t) {
            std::vector<u64> shape;
            for (u64 i = 0; i < t.ndim; ++i) shape.push_back(t.shape[i]);
            return shape;
        })
        .def_property_readonly("device", [](const Tensor &t) { return t.device; })
        .def_property_readonly("dtype", [](const Tensor &t) { return t.dtype; })
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_property_readonly("grad", [](Tensor &t) {
            return t.grad;
        }, py::return_value_policy::reference)
        .def_property_readonly("creator", [](Tensor &t) {
            return t.creator;
        }, py::return_value_policy::reference)
        .def("numpy", [](Tensor &t) {
            if (t.dtype != FLOAT32) throw std::runtime_error("Only Float32 supported for numpy() for now");
            
            u64 n = numel(&t);
            std::vector<float> host_data(n);
            
            if (t.device == CUDA) {
                arena_memcpy_d2h_cuda(host_data.data(), t.data, n * sizeof(float));
            } else {
                memcpy(host_data.data(), t.data, n * sizeof(float));
            }
            
            std::vector<ssize_t> shape;
            for (u64 i = 0; i < t.ndim; ++i) shape.push_back(t.shape[i]);
            
            return py::array_t<float>(shape, host_data.data());
        })
        .def("copy_from_numpy", [](Tensor &t, py::array_t<float> array) {
            auto buf = array.request();
            if (buf.size != numel(&t)) throw std::runtime_error("Size mismatch");
            
            if (t.device == CUDA) {
                arena_memcpy_h2d_cuda(t.data, buf.ptr, buf.size * sizeof(float));
            } else {
                memcpy(t.data, buf.ptr, buf.size * sizeof(float));
            }
        });

    m.def("tensor_init", [](Arena &meta, Arena &data, DEVICE device, DTYPE dtype,
                           std::vector<u64> shape, bool requires_grad) {
        return init(&meta, &data, device, dtype, shape.data(), shape.size(), requires_grad, NULL);
    }, py::return_value_policy::reference);

    m.def("numel", [](const Tensor &t) {
        return numel(&t);
    });

    m.def("zeros", [](Tensor &t) {
        zeros(&t, numel(&t));
    });

    m.def("set_ones_grad", [](Tensor &t) {
        set_ones_grad(&t);
    });

    // Node and Graph
    py::class_<Node>(m, "Node");

    m.def("create_node", [](Arena &meta, std::vector<Tensor*> inputs, Tensor &output, OP_TYPE op_type, u64 dim, bool keepdim) {
        // We need to copy pointers to a buffer compatible with C Tensor**
        Tensor **input_ptrs = (Tensor **)arena_alloc(&meta, inputs.size() * sizeof(Tensor*), 8);
        for (size_t i = 0; i < inputs.size(); ++i) input_ptrs[i] = inputs[i];
        
        return arena_node_alloc(&meta, input_ptrs, (int)inputs.size(), &output, get_op_impl(op_type), dim, keepdim);
    }, py::return_value_policy::reference);

    m.def("forward", [](Tensor &t) { if (t.creator) forward(t.creator); });
    m.def("backward", [](Tensor &t) { if (t.creator) backward(t.creator); });

    // Optimizers
    py::class_<SGD>(m, "SGD")
        .def(py::init([](float lr) {
            SGD sgd;
            sgd.lr = lr;
            return sgd;
        }));

    m.def("sgd_step_cuda", [](SGD &sgd, std::vector<Tensor*> params) {
        sgd_step_cuda(&sgd, params.data(), (int)params.size());
    });

    m.def("sgd_step_cpu", [](SGD &sgd, std::vector<Tensor*> params) {
        sgd_step_cpu(&sgd, params.data(), (int)params.size());
    });

    m.def("zero_grad_cuda", &zero_grad_cuda);
    m.def("zero_grad_cpu", &zero_grad_cpu);

    // Op Types
    py::enum_<OP_TYPE>(m, "OpType")
        .value("ADD", ADD)
        .value("SUB", SUB)
        .value("MUL", MUL)
        .value("DIV", DIV)
        .value("MATMUL", MATMUL)
        .value("ABS", ABS)
        .value("MEAN", MEAN)
        .export_values();
}
