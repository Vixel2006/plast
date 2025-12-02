#pragma once

#include "plast/core/shape_utils_cpp.h"
#include "plast/core/types.h"
#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"
#include "plast/core/shape_utils_cpp.h" // For core::broadcast_shapes

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric> // For std::iota

namespace plast
{
namespace ops
{

class MatmulOperation : public BaseOperation
{
  public:
    const std::string& name() const override
    {
        static std::string op_name = "matmul";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        if (input_shapes.size() != 2)
        {
            throw std::runtime_error("Matmul operation requires exactly two input tensors.");
        }

        const std::vector<size_t>& lhs_shape = input_shapes[0];
        const std::vector<size_t>& rhs_shape = input_shapes[1];

        size_t lhs_ndim = lhs_shape.size();
        size_t rhs_ndim = rhs_shape.size();

        if (lhs_ndim < 1 || rhs_ndim < 1)
        {
            throw std::runtime_error("Matmul operation can't be done on a scalar.");
        }

        // Handle vector-matrix, matrix-vector, vector-vector cases
        // If either is 1D, treat as (1, D) or (D, 1)
        std::vector<size_t> effective_lhs_shape = lhs_shape;
        std::vector<size_t> effective_rhs_shape = rhs_shape;

        if (lhs_ndim == 1)
        {
            effective_lhs_shape.insert(effective_lhs_shape.begin(), 1); // (D) -> (1, D)
            lhs_ndim++;
        }
        if (rhs_ndim == 1)
        {
            effective_rhs_shape.push_back(1); // (D) -> (D, 1)
            rhs_ndim++;
        }

        if (lhs_ndim < 2 || rhs_ndim < 2)
        {
            throw std::runtime_error("Matmul operation requires inputs with at least 2 dimensions "
                                     "after handling 1D tensors.");
        }

        // Extract batch dimensions
        std::vector<size_t> lhs_batch_shape(effective_lhs_shape.begin(),
                                            effective_lhs_shape.end() - 2);
        std::vector<size_t> rhs_batch_shape(effective_rhs_shape.begin(),
                                            effective_rhs_shape.end() - 2);

        // Broadcast batch dimensions
        std::vector<size_t> output_batch_shape =
            core::broadcast_shapes(lhs_batch_shape, rhs_batch_shape);

        // Extract N, K, M
        size_t N = effective_lhs_shape[lhs_ndim - 2];
        size_t K1 = effective_lhs_shape[lhs_ndim - 1];
        size_t K2 = effective_rhs_shape[rhs_ndim - 2];
        size_t M = effective_rhs_shape[rhs_ndim - 1];

        if (K1 != K2)
        {
            throw std::runtime_error("Matmul operation: K dimensions do not match.");
        }

        // Construct final output shape
        std::vector<size_t> output_shape = output_batch_shape;
        output_shape.push_back(N);
        output_shape.push_back(M);

        // If original inputs were 1D, adjust output shape back
        if (lhs_shape.size() == 1 && rhs_shape.size() == 1)
        {
            // Vector-vector dot product results in scalar
            return {}; // Empty vector for scalar
        }
        else if (lhs_shape.size() == 1)
        {
            // Vector-matrix product, output is (M)
            output_shape.erase(output_shape.begin()); // Remove the prepended 1
        }
        else if (rhs_shape.size() == 1)
        {
            // Matrix-vector product, output is (N)
            output_shape.pop_back(); // Remove the appended 1
        }

        return output_shape;
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;
};

} // namespace ops
} // namespace plast
