#include <memory>
#include <pybind11/numpy.h>     // For py::array
#include <pybind11/operators.h> // For operator overloading
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, etc.

#include "plast/core/types.h"
#include "plast/execution/engine.h"
#include "plast/graph/node.h"
#include "plast/ops/base_op.h"
#include "plast/ops/binary/add.h"    // Example operation
#include "plast/ops/binary/sub.h"    // Include for SubOperation
#include "plast/ops/binary/mul.h"    // Include for MulOperation
#include "plast/ops/init/init_ops.h" // Include for new C++ initialization operations
#include "plast/tensor/tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(_plast_cpp_core, m)
{
    m.doc() = "plast C++ Core Bindings";

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
    py::class_<plast::tensor::Tensor>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&, plast::core::DType, plast::core::DeviceType>(),
             py::arg("shape"), py::arg("dtype"), py::arg("device"))
        .def_property_readonly("shape", &plast::tensor::Tensor::shape)
        .def_property_readonly("dtype", &plast::tensor::Tensor::dtype)
        .def_property_readonly("device", &plast::tensor::Tensor::device)
        .def("to", &plast::tensor::Tensor::to, py::arg("target_device"))
        .def("clone", &plast::tensor::Tensor::clone)
        .def("num_elements", &plast::tensor::Tensor::num_elements)
        .def("nbytes", &plast::tensor::Tensor::nbytes)
        .def("data_ptr", &plast::tensor::Tensor::data) // Expose the raw data pointer
        .def("_get_data_as_numpy",
             [](plast::tensor::Tensor& t) -> py::array
             {
                 // Determine numpy dtype based on plast::core::DType
                 py::dtype numpy_dtype;
                 size_t itemsize;
                 switch (t.dtype())
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
                 std::vector<py::ssize_t> strides(t.shape().size());
                 py::ssize_t current_stride = itemsize;
                 for (int i = t.shape().size() - 1; i >= 0; --i)
                 {
                     strides[i] = current_stride;
                     current_stride *= t.shape()[i];
                 }

                 return py::array(numpy_dtype, t.shape(), strides, t.data());
             })
        .def("__repr__",
             [](const plast::tensor::Tensor& t)
             {
                 std::stringstream ss;
                 ss << "Tensor(shape=" << py::cast(t.shape()) << ", dtype=" << py::cast(t.dtype())
                    << ", device=" << py::cast(t.device()) << ")";
                 return ss.str();
             });

    // Bind Node class (using std::shared_ptr for ownership management)
    py::class_<plast::graph::Node, std::shared_ptr<plast::graph::Node>>(m, "Node")
        .def(py::init<const plast::tensor::Tensor&>(), py::arg("value")) // For leaf nodes
        .def_property_readonly("is_leaf", &plast::graph::Node::is_leaf)
        .def_property_readonly("operation", &plast::graph::Node::operation)
        .def_property_readonly("inputs", &plast::graph::Node::inputs)
        .def_property_readonly("shape", &plast::graph::Node::shape) // Expose the shape property
        .def("has_cached_value", &plast::graph::Node::has_cached_value)
        .def("get_cached_value", &plast::graph::Node::get_cached_value);

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

    // Bind ExecutionEngine class
    py::class_<plast::execution::ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<>())
        .def("execute", &plast::execution::ExecutionEngine::execute, py::arg("root_node"))
        .def("clear_cache", &plast::execution::ExecutionEngine::clear_cache);

    // Bind initialization functions
    m.def("zeros", &plast::ops::init::zeros, py::arg("shape"), py::arg("dtype"), py::arg("device"));
    m.def("ones", &plast::ops::init::ones, py::arg("shape"), py::arg("dtype"), py::arg("device"));
    m.def("randn", &plast::ops::init::randn, py::arg("shape"), py::arg("dtype"), py::arg("device"),
          py::arg("seed"));
    m.def("uniform", &plast::ops::init::uniform, py::arg("shape"), py::arg("dtype"),
          py::arg("device"), py::arg("low"), py::arg("high"));
    m.def(
        "from_data",
        [](py::array data_array, const std::vector<size_t>& shape, plast::core::DType dtype,
           plast::core::DeviceType device)
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
}
