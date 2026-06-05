#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
extern "C" {
#include "core/arena.h"
#ifdef CUDA_AVAILABLE
#include "core/arena_cuda.h"
#endif
#include "core/graph.h"
#include "core/node.h"
#include "core/op.h"
#include "optimizers/adam.h"
#include "optimizers/adamw.h"
#include "optimizers/sgd.h"
#include "optimizers/zero_grad.h"
#include "core/tensor.h"
}

namespace py = pybind11;

// Wrapper for DTYPE
enum PyDType { INT32_PY = INT32, FLOAT32_PY = FLOAT32 };

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
      .def_property_readonly("shape",
                             [](const Tensor &t) {
                               std::vector<u64> shape;
                               for (u64 i = 0; i < t.ndim; ++i)
                                 shape.push_back(t.shape[i]);
                               return shape;
                             })
      .def_property_readonly("device", [](const Tensor &t) { return t.device; })
      .def_property_readonly("dtype", [](const Tensor &t) { return t.dtype; })
      .def_readwrite("requires_grad", &Tensor::requires_grad)
      .def_property_readonly(
          "grad", [](Tensor &t) { return t.grad; },
          py::return_value_policy::reference)
      .def_property_readonly(
          "creator", [](Tensor &t) { return t.creator; },
          py::return_value_policy::reference)
      .def("numpy",
           [](Tensor &t) {
             if (t.dtype != FLOAT32)
               throw std::runtime_error(
                   "Only Float32 supported for numpy() for now");

             u64 n = numel(&t);
             std::vector<float> host_data(n);

             if (t.device == CUDA) {
#ifdef CUDA_AVAILABLE
               arena_memcpy_d2h_cuda(host_data.data(), t.data,
                                     n * sizeof(float));
#else
                throw std::runtime_error("CUDA not available in this build");
#endif
             } else {
               memcpy(host_data.data(), t.data, n * sizeof(float));
             }

             std::vector<ssize_t> shape;
             for (u64 i = 0; i < t.ndim; ++i)
               shape.push_back(t.shape[i]);

             return py::array_t<float>(shape, host_data.data());
           })
      .def_property_readonly("strides",
                             [](const Tensor &t) {
                               std::vector<u64> strides;
                               for (u64 i = 0; i < t.ndim; ++i)
                                 strides.push_back(t.strides[i]);
                               return strides;
                             })
      .def_property_readonly("is_contiguous",
                             [](const Tensor &t) { return is_contiguous(&t); })
      .def("copy_from_numpy", [](Tensor &t, py::array_t<float> array) {
        auto buf = array.request();
        if (buf.size != numel(&t))
          throw std::runtime_error("Size mismatch");

        if (t.device == CUDA) {
#ifdef CUDA_AVAILABLE
          arena_memcpy_h2d_cuda(t.data, buf.ptr, buf.size * sizeof(float));
#else
                throw std::runtime_error("CUDA not available in this build");
#endif
        } else {
          memcpy(t.data, buf.ptr, buf.size * sizeof(float));
        }
      });

  m.def(
      "tensor_init",
      [](Arena &meta, Arena &data, DEVICE device, DTYPE dtype,
         std::vector<u64> shape, bool requires_grad) {
        return init(&meta, &data, device, dtype, shape.data(), shape.size(),
                    requires_grad, NULL);
      },
      py::return_value_policy::reference);

  m.def("numel", [](const Tensor &t) { return numel(&t); });

  m.def("zeros", [](Tensor &t) { zeros(&t, numel(&t)); });

  m.def("set_ones_grad", [](Tensor &t) { set_ones_grad(&t); });

  m.def("ones", [](Tensor &t) { ones(&t, numel(&t)); });

  // Node and Graph
  py::class_<Node>(m, "Node");

  m.def(
      "create_node",
      [](Arena &meta, std::vector<Tensor *> inputs, Tensor &output,
         OP_TYPE op_type, u64 dim, u64 keepdim, float fval) {
        Tensor **input_ptrs =
            (Tensor **)arena_alloc(&meta, inputs.size() * sizeof(Tensor *), 8);
        for (size_t i = 0; i < inputs.size(); ++i)
          input_ptrs[i] = inputs[i];

        KernelParams params = {dim, keepdim, fval};
        if (op_type == CONV2D) {
          params.dim = (u64)&meta;
        }
        return arena_node_alloc(&meta, input_ptrs, (int)inputs.size(), &output,
                                get_op_impl(op_type), params);
      },
      py::return_value_policy::reference);

  m.def("forward", [](Tensor &t) {
    if (t.creator)
      forward(t.creator);
  });
  m.def("backward", [](Tensor &t) {
    if (t.creator)
      backward(t.creator);
  });

  // Optimizers
  py::class_<SGD>(m, "SGD")
      .def(py::init([](float lr) {
        SGD sgd;
        sgd.lr = lr;
        return sgd;
      }))
      .def_readwrite("lr", &SGD::lr);

#ifdef CUDA_AVAILABLE
  m.def("sgd_step_cuda", [](SGD &sgd, std::vector<Tensor *> params) {
    sgd_step_cuda(&sgd, params.data(), (int)params.size());
  });
#endif

  m.def("sgd_step_cpu", [](SGD &sgd, std::vector<Tensor *> params) {
    sgd_step_cpu(&sgd, params.data(), (int)params.size());
  });

  // Adam Optimizer Bindings
  py::class_<Adam>(m, "Adam")
      .def(py::init([](Arena &optimizer_arena, Arena &data_arena, float lr,
                       float beta1, float beta2, float epsilon) {
        return alloc_adam(&optimizer_arena, &data_arena, lr, beta1, beta2,
                          epsilon);
      }))
      .def_readwrite("lr", &Adam::lr)
      .def_readwrite("beta1", &Adam::beta1)
      .def_readwrite("beta2", &Adam::beta2)
      .def_readwrite("epsilon", &Adam::epsilon)
      .def_readwrite("t", &Adam::t);

  m.def("adam_step_cpu", [](Adam &adam, std::vector<Tensor *> params) {
    adam_step_cpu(&adam, params.data(), (int)params.size());
  });

#ifdef CUDA_AVAILABLE
  m.def("adam_step_cuda", [](Adam &adam, std::vector<Tensor *> params) {
    adam_step_cuda(&adam, params.data(), (int)params.size());
  });
#endif

  // AdamW Optimizer Bindings
  py::class_<AdamW>(m, "AdamW")
      .def(py::init([](Arena &optimizer_arena, Arena &data_arena, float lr,
                       float beta1, float beta2, float epsilon,
                       float weight_decay) {
        return adamw_alloc(&optimizer_arena, &data_arena, lr, beta1, beta2,
                           epsilon, weight_decay);
      }))
      .def_readwrite("lr", &AdamW::lr)
      .def_readwrite("beta1", &AdamW::beta1)
      .def_readwrite("beta2", &AdamW::beta2)
      .def_readwrite("epsilon", &AdamW::epsilon)
      .def_readwrite("weight_decay", &AdamW::weight_decay)
      .def_readwrite("t", &AdamW::t);

  m.def("adamw_step_cpu", [](AdamW &adamw, std::vector<Tensor *> params) {
    adamw_step_cpu(&adamw, params.data(), (int)params.size());
  });

#ifdef CUDA_AVAILABLE
  m.def("adamw_step_cuda", [](AdamW &adamw, std::vector<Tensor *> params) {
    adamw_step_cuda(&adamw, params.data(), (int)params.size());
  });
#endif

#ifdef CUDA_AVAILABLE
  m.def("zero_grad_cuda", &zero_grad_cuda);
#endif
  m.def("zero_grad_cpu", &zero_grad_cpu);

  // Op Types
  py::enum_<OP_TYPE>(m, "OpType")
      .value("ADD", ADD)
      .value("SUB", SUB)
      .value("MUL", MUL)
      .value("DIV", DIV)
      .value("MATMUL", MATMUL)
      .value("LEAKY_RELU", LEAKY_RELU)
      .value("LOG", LOG)
      .value("EXP", EXP)
      .value("ABS", ABS)
      .value("NEG", NEG)
      .value("SIN", SIN)
      .value("COS", COS)
      .value("TAN", TAN)
      .value("VIEW", VIEW)
      .value("TRANSPOSE", TRANSPOSE)
      .value("UNSQUEEZE", UNSQUEEZE)
      .value("SQUEEZE", SQUEEZE)
      .value("EXPAND", EXPAND)
      .value("BROADCAST", BROADCAST)
      .value("MEAN", MEAN)
      .value("MIN", MIN)
      .value("MAX", MAX)
      .value("SUM", SUM)
      .value("FLATTEN", FLATTEN)
      .value("CONV2D", CONV2D)
      .export_values();
}
