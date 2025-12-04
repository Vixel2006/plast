#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "plast/core/device_management.h"
#include "plast/core/types.h"
#include "plast/execution/engine.h"
#include "plast/graph/node.h"
#include "plast/ops/base_op.h"
#include "plast/ops/binary/add.h"
#include "plast/ops/binary/matmul.h"
#include "plast/ops/binary/mul.h"
#include "plast/ops/binary/sub.h"
#include "plast/ops/init/init_ops.h"
#include "plast/ops/movement/broadcast.h"
#include "plast/ops/movement/expand.h"
#include "plast/ops/movement/squeeze.h"
#include "plast/ops/movement/transpose.h"
#include "plast/ops/movement/unsqueeze.h"
#include "plast/ops/movement/view.h"
#include "plast/ops/reduction/max.h"
#include "plast/ops/reduction/mean.h"
#include "plast/ops/reduction/min.h"
#include "plast/ops/reduction/sum.h"
#include "plast/ops/unary/abs.h"
#include "plast/ops/unary/exp.h"
#include "plast/ops/unary/leaky_relu.h"
#include "plast/ops/unary/log.h"
#include "plast/ops/unary/relu.h"
#include "plast/tensor/tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(_plast_cpp_core, m)
{
    m.doc() = "plast C++ Core Bindings";

    // Expose is_cuda_available function
    m.def("is_cuda_available", &plast::core::is_cuda_available, "Checks if CUDA is available.");

    // Bind DType enum
    py::enum_<plast::core::DType>(m, "DType")
        .value("UNKNOWN", plast::core::DType::UNKNOWN)
        .value("FLOAT32", plast::core::DType::FLOAT32)
        .value("FLOAT64", plast::core::DType::FLOAT64)
        .value("INT8", plast::core::DType::INT8)
        .value("INT16", plast::core::DType::INT16)
        .value("INT32", plast::core::DType::INT32)
        .value("INT64", plast::core::DType::INT64)
        .value("UINT8", plast::core::DType::UINT8)
        .value("UINT16", plast::core::DType::UINT16)
        .value("UINT32", plast::core::DType::UINT32)
        .value("UINT64", plast::core::DType::UINT64)
        .value("BOOL", plast::core::DType::BOOL)
        .export_values();

    // Bind DeviceType enum
    py::enum_<plast::core::DeviceType>(m, "DeviceType")
        .value("CPU", plast::core::DeviceType::CPU)
        .value("CUDA", plast::core::DeviceType::CUDA)
        .export_values();

    // Bind Tensor class
    py::class_<plast::tensor::Tensor, std::shared_ptr<plast::tensor::Tensor>>(m, "Tensor")
        .def(py::init([](const std::vector<size_t>& shape, plast::core::DType dtype,
                         plast::core::DeviceType device)
                      { return std::make_shared<plast::tensor::Tensor>(shape, dtype, device); }),
             py::arg("shape"), py::arg("dtype"), py::arg("device"))
        .def_property_readonly("shape", &plast::tensor::Tensor::shape)
        .def_property_readonly("strides", &plast::tensor::Tensor::strides)
        .def_property_readonly("dtype", &plast::tensor::Tensor::dtype)
        .def_property_readonly("device", &plast::tensor::Tensor::device)
        .def("to", &plast::tensor::Tensor::to, py::arg("target_device"))
        .def("clone", &plast::tensor::Tensor::clone)
        .def("reshape",
             py::overload_cast<const std::vector<size_t>&>(&plast::tensor::Tensor::reshape,
                                                           py::const_),
             py::arg("new_shape"))
        .def("reshape",
             py::overload_cast<const std::vector<size_t>&, const std::vector<size_t>&>(
                 &plast::tensor::Tensor::reshape, py::const_),
             py::arg("new_shape"), py::arg("new_strides"))
        .def("num_elements", &plast::tensor::Tensor::num_elements)
        .def("nbytes", &plast::tensor::Tensor::nbytes)
        .def("data_ptr", &plast::tensor::Tensor::data) // Expose the raw data pointer
        .def("_get_data_as_numpy",
             [](plast::tensor::Tensor& t) -> py::array
             {
                 // If the tensor is on CUDA, transfer it to CPU first
                 plast::tensor::Tensor cpu_tensor = (t.device() == plast::core::DeviceType::CUDA)
                                                        ? t.to(plast::core::DeviceType::CPU)
                                                        : t.clone();

                 // Determine numpy dtype based on plast::core::DType
                 py::dtype numpy_dtype;
                 size_t itemsize;
                 switch (cpu_tensor.dtype())
                 {
                 case plast::core::DType::FLOAT32:
                     numpy_dtype = py::dtype::of<float>();
                     itemsize = sizeof(float);
                     break;
                 case plast::core::DType::FLOAT64:
                     numpy_dtype = py::dtype::of<double>();
                     itemsize = sizeof(double);
                     break;
                 case plast::core::DType::INT8:
                     numpy_dtype = py::dtype::of<int8_t>();
                     itemsize = sizeof(int8_t);
                     break;
                 case plast::core::DType::INT16:
                     numpy_dtype = py::dtype::of<int16_t>();
                     itemsize = sizeof(int16_t);
                     break;
                 case plast::core::DType::INT32:
                     numpy_dtype = py::dtype::of<int32_t>();
                     itemsize = sizeof(int32_t);
                     break;
                 case plast::core::DType::INT64:
                     numpy_dtype = py::dtype::of<int64_t>();
                     itemsize = sizeof(int64_t);
                     break;
                 case plast::core::DType::UINT8:
                     numpy_dtype = py::dtype::of<uint8_t>();
                     itemsize = sizeof(uint8_t);
                     break;
                 case plast::core::DType::UINT16:
                     numpy_dtype = py::dtype::of<uint16_t>();
                     itemsize = sizeof(uint16_t);
                     break;
                 case plast::core::DType::UINT32:
                     numpy_dtype = py::dtype::of<uint32_t>();
                     itemsize = sizeof(uint32_t);
                     break;
                 case plast::core::DType::UINT64:
                     numpy_dtype = py::dtype::of<uint64_t>();
                     itemsize = sizeof(uint64_t);
                     break;
                 case plast::core::DType::BOOL:
                     numpy_dtype = py::dtype::of<bool>();
                     itemsize = sizeof(bool);
                     break;
                 default:
                     throw std::runtime_error("Unsupported DType for numpy conversion.");
                 }

                 // Calculate strides
                 std::vector<py::ssize_t> strides_bytes(cpu_tensor.strides().size());
                 for (size_t i = 0; i < cpu_tensor.strides().size(); ++i)
                 {
                     strides_bytes[i] = cpu_tensor.strides()[i] * itemsize;
                 }

                 return py::array(numpy_dtype, cpu_tensor.shape(), strides_bytes,
                                  cpu_tensor.data());
             })
        .def("__repr__",
             [](const plast::tensor::Tensor& t)
             {
                 std::stringstream ss;
                 ss << "Tensor(shape=" << py::cast(t.shape())
                    << ", strides=" << py::cast(t.strides()) << ", dtype=" << py::cast(t.dtype())
                    << ", device=" << py::cast(t.device()) << ")";
                 return ss.str();
             });

    // Bind Node class (using std::shared_ptr for ownership management)
    py::class_<plast::graph::Node, std::shared_ptr<plast::graph::Node>>(m, "Node")
        .def(py::init([](std::shared_ptr<plast::tensor::Tensor> value)
                      { return std::make_shared<plast::graph::Node>(value); }),
             py::arg("value")) // For leaf nodes, takes shared_ptr to Tensor
        .def_property_readonly("is_leaf", &plast::graph::Node::is_leaf)
        .def_property_readonly("operation", &plast::graph::Node::operation)
        .def_property_readonly("inputs", &plast::graph::Node::inputs)
        .def_property_readonly("shape", &plast::graph::Node::shape) // Expose the shape property
        .def("has_output_tensor", &plast::graph::Node::has_output_tensor)
        .def("get_output_tensor", &plast::graph::Node::get_output_tensor);

    // Bind BaseOperation abstract class
    py::class_<plast::ops::BaseOperation, std::shared_ptr<plast::ops::BaseOperation>>(
        m, "BaseOperation")
        .def_property_readonly("name", &plast::ops::BaseOperation::name);

    // Bind specific operations (e.g., AddOperation, SubOperation)
    py::class_<plast::ops::AddOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::AddOperation>>(m, "AddOperation")
        .def(py::init<>());

    py::class_<plast::ops::SubOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::SubOperation>>(m, "SubOperation")
        .def(py::init<>());

    py::class_<plast::ops::MulOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::MulOperation>>(m, "MulOperation")
        .def(py::init<>());

    py::class_<plast::ops::AbsOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::AbsOperation>>(m, "AbsOperation")
        .def(py::init<>());

    py::class_<plast::ops::LeakyReluOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::LeakyReluOperation>>(m, "LeakyReluOperation")
        .def(py::init<float>(), py::arg("alpha") = 0.01f);

    py::class_<plast::ops::ViewOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::ViewOperation>>(m, "ViewOperation")
        .def(py::init<const std::vector<size_t>&>(), py::arg("new_shape"));

    py::class_<plast::ops::TransposeOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::TransposeOperation>>(m, "TransposeOperation")
        .def(py::init<size_t, size_t>(), py::arg("N"), py::arg("M"));

    py::class_<plast::ops::SqueezeOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::SqueezeOperation>>(m, "SqueezeOperation")
        .def(py::init<size_t, size_t>(), py::arg("N"), py::arg("M"));

    py::class_<plast::ops::UnsqueezeOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::UnsqueezeOperation>>(m, "UnsqueezeOperation")
        .def(py::init<size_t>(), py::arg("dim"));

    py::class_<plast::ops::ExpandOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::ExpandOperation>>(m, "ExpandOperation")
        .def(py::init<const std::vector<size_t>&>(), py::arg("new_shape"));

    py::class_<plast::ops::BroadcastOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::BroadcastOperation>>(m, "BroadcastOperation")
        .def(py::init<const std::vector<size_t>&>(), py::arg("target_shape"));

    py::class_<plast::ops::MinOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::MinOperation>>(m, "MinOperation")
        .def(py::init<int, bool>(), py::arg("dim"), py::arg("keepdim"))
        .def(py::init<bool>(), py::arg("full_reduction"));

    py::class_<plast::ops::MeanOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::MeanOperation>>(m, "MeanOperation")
        .def(py::init<int, bool>(), py::arg("dim"), py::arg("keepdim"))
        .def(py::init<bool>(), py::arg("full_reduction"));

    py::class_<plast::ops::SumOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::SumOperation>>(m, "SumOperation")
        .def(py::init<int, bool>(), py::arg("dim"), py::arg("keepdim"))
        .def(py::init<bool>(), py::arg("full_reduction"));

    py::class_<plast::ops::MaxOperation, plast::ops::BaseOperation,
               std::shared_ptr<plast::ops::MaxOperation>>(m, "MaxOperation")
        .def(py::init<int, bool>(), py::arg("dim"), py::arg("keepdim"))
        .def(py::init<bool>(), py::arg("full_reduction"));

    // Bind ExecutionEngine class
    py::class_<plast::execution::ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<>())
        .def("execute", &plast::execution::ExecutionEngine::execute);

    // Bind initialization functions
    m.def(
        "zeros",
        [](const std::vector<size_t>& shape, plast::core::DType dtype,
           plast::core::DeviceType device)
        { return plast::ops::init::zeros(shape, dtype, device); },
        py::arg("shape"), py::arg("dtype"), py::arg("device"));
    m.def(
        "ones",
        [](const std::vector<size_t>& shape, plast::core::DType dtype,
           plast::core::DeviceType device) { return plast::ops::init::ones(shape, dtype, device); },
        py::arg("shape"), py::arg("dtype"), py::arg("device"));
    m.def(
        "randn",
        [](const std::vector<size_t>& shape, plast::core::DType dtype,
           plast::core::DeviceType device, int seed)
        { return plast::ops::init::randn(shape, dtype, device, seed); },
        py::arg("shape"), py::arg("dtype"), py::arg("device"), py::arg("seed"));
    m.def(
        "uniform",
        [](const std::vector<size_t>& shape, plast::core::DType dtype,
           plast::core::DeviceType device, float low, float high, int seed)
        { return plast::ops::init::uniform(shape, dtype, device, low, high, seed); },
        py::arg("shape"), py::arg("dtype"), py::arg("device"), py::arg("low"), py::arg("high"),
        py::arg("seed"));
    m.def(
        "from_data",
        [](py::array data_array, const std::vector<size_t>& shape, plast::core::DType dtype,
           plast::core::DeviceType device) -> std::shared_ptr<plast::tensor::Tensor>
        {
            // Ensure the numpy array is contiguous and of the correct type
            py::buffer_info buf_info = data_array.request();
            if (buf_info.format != py::format_descriptor<float>::format() &&
                buf_info.format != py::format_descriptor<int>::format())
            {
                throw std::runtime_error(
                    "Unsupported data type for from_data. Only float and int are supported.");
            }
            return plast::ops::init::from_data(buf_info.ptr, shape, dtype, device);
        },
        py::arg("data"), py::arg("shape"), py::arg("dtype"), py::arg("device"));

    // Example of how Python ops would create C++ graph nodes
    // This function would be called from Python's `plast.ops.add`
    m.def(
        "add_op_node",
        [](std::shared_ptr<plast::graph::Node> lhs, std::shared_ptr<plast::graph::Node> rhs)
        {
            auto op = std::make_shared<plast::ops::AddOperation>();
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{lhs, rhs});
        },
        py::arg("lhs"), py::arg("rhs"));

    // This function would be called from Python's `plast.ops.sub`
    m.def(
        "sub_op_node",
        [](std::shared_ptr<plast::graph::Node> lhs, std::shared_ptr<plast::graph::Node> rhs)
        {
            auto op = std::make_shared<plast::ops::SubOperation>();
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{lhs, rhs});
        },
        py::arg("lhs"), py::arg("rhs"));

    // This function would be called from python's `plast.ops.mul`
    m.def(
        "mul_op_node",
        [](std::shared_ptr<plast::graph::Node> lhs, std::shared_ptr<plast::graph::Node> rhs)
        {
            auto op = std::make_shared<plast::ops::MulOperation>();
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{lhs, rhs});
        },
        py::arg("lhs"), py::arg("rhs"));

    // This function would be called from python's `plast.ops.matmul`
    m.def(
        "matmul_op_node",
        [](std::shared_ptr<plast::graph::Node> lhs, std::shared_ptr<plast::graph::Node> rhs)
        {
            auto op = std::make_shared<plast::ops::MatmulOperation>();
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{lhs, rhs});
        },
        py::arg("lhs"), py::arg("rhs"));

    // This function would be called from python's `plast.ops.abs`
    m.def(
        "abs_op_node",
        [](std::shared_ptr<plast::graph::Node> input)
        {
            auto op = std::make_shared<plast::ops::AbsOperation>();
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"));

    // This function would be called from python's `plast.ops.relu`
    m.def(
        "relu_op_node",
        [](std::shared_ptr<plast::graph::Node> input)
        {
            auto op = std::make_shared<plast::ops::ReluOperation>();
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"));

    // This function would be called from python's `plast.ops.leaky_relu`
    m.def(
        "leaky_relu_op_node",
        [](std::shared_ptr<plast::graph::Node> input, float alpha)
        {
            auto op = std::make_shared<plast::ops::LeakyReluOperation>(alpha);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("alpha") = 0.01f);

    m.def(
        "exp_op_node",
        [](std::shared_ptr<plast::graph::Node> input)
        {
            auto op = std::make_shared<plast::ops::ExpOperation>();
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"));

    m.def(
        "log_op_node",
        [](std::shared_ptr<plast::graph::Node> input)
        {
            auto op = std::make_shared<plast::ops::LogOperation>();
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"));

    // This function would be called from python's `plast.ops.view`
    m.def(
        "view_op_node",
        [](std::shared_ptr<plast::graph::Node> input, const std::vector<size_t>& new_shape)
        {
            auto op = std::make_shared<plast::ops::ViewOperation>(new_shape);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("new_shape"));

    // This function would be called from python's `plast.ops.transpose`
    m.def(
        "transpose_op_node",
        [](std::shared_ptr<plast::graph::Node> input, size_t N, size_t M)
        {
            auto op = std::make_shared<plast::ops::TransposeOperation>(N, M);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("N"), py::arg("M"));

    // This function would be called from python's `plast.ops.squeeze`
    m.def(
        "squeeze_op_node",
        [](std::shared_ptr<plast::graph::Node> input, size_t N, size_t M)
        {
            auto op = std::make_shared<plast::ops::SqueezeOperation>(N, M);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("N"), py::arg("M"));

    // This function would be called from python's `plast.ops.unsqueeze`
    m.def(
        "unsqueeze_op_node",
        [](std::shared_ptr<plast::graph::Node> input, size_t dim)
        {
            auto op = std::make_shared<plast::ops::UnsqueezeOperation>(dim);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("dim"));

    // This function would be called from python's `plast.ops.expand`
    m.def(
        "expand_op_node",
        [](std::shared_ptr<plast::graph::Node> input, const std::vector<size_t>& new_shape)
        {
            auto op = std::make_shared<plast::ops::ExpandOperation>(new_shape);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("new_shape"));

    // This function would be called from python's `plast.ops.broadcast_to`
    m.def(
        "broadcast_op_node",
        [](std::shared_ptr<plast::graph::Node> input, const std::vector<size_t>& target_shape)
        {
            auto op = std::make_shared<plast::ops::BroadcastOperation>(target_shape);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("target_shape"));

    // Min operations
    m.def(
        "min_op_node_full",
        [](std::shared_ptr<plast::graph::Node> input)
        {
            auto op = std::make_shared<plast::ops::MinOperation>(true); // full_reduction = true
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"));

    m.def(
        "min_op_node_dim",
        [](std::shared_ptr<plast::graph::Node> input, int dim, bool keepdim)
        {
            auto op = std::make_shared<plast::ops::MinOperation>(dim, keepdim);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("dim"), py::arg("keepdim"));

    // Mean operations
    m.def(
        "mean_op_node_full",
        [](std::shared_ptr<plast::graph::Node> input)
        {
            auto op = std::make_shared<plast::ops::MeanOperation>(true); // full_reduction = true
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"));

    m.def(
        "mean_op_node_dim",
        [](std::shared_ptr<plast::graph::Node> input, int dim, bool keepdim)
        {
            auto op = std::make_shared<plast::ops::MeanOperation>(dim, keepdim);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("dim"), py::arg("keepdim"));

    // Sum operations
    m.def(
        "sum_op_node_full",
        [](std::shared_ptr<plast::graph::Node> input)
        {
            auto op = std::make_shared<plast::ops::SumOperation>(true); // full_reduction = true
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"));

    m.def(
        "sum_op_node_dim",
        [](std::shared_ptr<plast::graph::Node> input, int dim, bool keepdim)
        {
            auto op = std::make_shared<plast::ops::SumOperation>(dim, keepdim);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("dim"), py::arg("keepdim"));

    // Max operations
    m.def(
        "max_op_node_full",
        [](std::shared_ptr<plast::graph::Node> input)
        {
            auto op = std::make_shared<plast::ops::MaxOperation>(true); // full_reduction = true
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"));

    m.def(
        "max_op_node_dim",
        [](std::shared_ptr<plast::graph::Node> input, int dim, bool keepdim)
        {
            auto op = std::make_shared<plast::ops::MaxOperation>(dim, keepdim);
            return std::make_shared<plast::graph::Node>(
                op, std::vector<std::shared_ptr<plast::graph::Node>>{input});
        },
        py::arg("input"), py::arg("dim"), py::arg("keepdim"));
}
