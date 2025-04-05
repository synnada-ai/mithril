// Copyright 2022 Synnada, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "stdio.h"
#include "ops.h"
#include "utils.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

float add_lambda(float x, float y)
{
    return x + y;
}

float multiply_lambda(float x, float y)
{
    return x * y;
}

float subtract_lambda(float x, float y)
{
    return x - y;
}

static float squared_diff(float x, float y) {
    float diff = x - y;
    return diff * diff;
}

void add(Array *output, Array *left, Array *right)
{
    binary_array_iterator(left, right, output, add_lambda);
}

void multiplication(Array *output, Array *left, Array *right)
{
    binary_array_iterator(left, right, output, multiply_lambda);
}

void subtract(Array *output, Array *left, Array *right)
{
    binary_array_iterator(left, right, output, subtract_lambda);
}

void transpose(Array *output, const Array *input, void *axes)
{
    if (!axes) {
        // Default 2D transpose
        // TODO: Currently only support 2D, ND support should be added.
        int m = input->shape[0], n = input->shape[1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                size_t input_idx = i * input->strides[0] + j * input->strides[1];
                size_t output_idx = j * output->strides[0] + i * output->strides[1];
                output->data[output_idx] = input->data[input_idx];
            }
        }
    } else {
        // General N-D transpose
        int *axes_arr = (int *)axes;
        int *inv_axes = malloc(input->ndim * sizeof(int));
        invert_permutation(axes_arr, inv_axes, input->ndim);

        // Recompute output strides based on transposed axes
        output->strides = compute_strides(output->shape, output->ndim);

        // Copy data using inverse axes
        for (size_t i=0; i<input->size; i++) {
            size_t out_idx = 0;
            int tmp = i;
            for (int d=input->ndim-1; d>=0; d--) {
                int coord = tmp % input->shape[d];
                tmp /= input->shape[d];
                out_idx += coord * output->strides[inv_axes[d]];
            }
            output->data[out_idx] = input->data[i];
        }
        free(inv_axes);
    }
}

void matrix_multiplication(Array *output, const Array *left, const Array *right) 
{
    int max_ndim = MAX(left->ndim, right->ndim);
    int *lshape = pad_shape(left, max_ndim);
    int *rshape = pad_shape(right, max_ndim);

    int *out_shape = (int*)malloc(max_ndim * sizeof(int));
    for(int i=0; i<max_ndim-2; i++)
        out_shape[i] = MAX(lshape[i], rshape[i]);
    out_shape[max_ndim-2] = lshape[max_ndim-2];  // M
    out_shape[max_ndim-1] = rshape[max_ndim-1];  // N
    const int M = out_shape[max_ndim-2];
    const int N = out_shape[max_ndim-1];
    const int K = lshape[max_ndim-1];  // Inner dimension

    output->strides = compute_strides(out_shape, max_ndim);
    int *lstrides = compute_strides(lshape, max_ndim);
    int *rstrides = compute_strides(rshape, max_ndim);

    if (K != rshape[max_ndim-2]) {
        printf("Dimension mismatch: %d vs %d\n", K, rshape[max_ndim-2]);
        exit(1);
    }

    size_t batch_size = 1;
    for (int i=0; i<max_ndim-2; i++)
        batch_size *= MAX(lshape[i], rshape[i]);

    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; b++) {
        size_t l_offset = loc(b, lshape, lstrides, max_ndim-2);
        size_t r_offset = loc(b, rshape, rstrides, max_ndim-2);

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    size_t left_idx = l_offset + i*lstrides[max_ndim-2] + k*lstrides[max_ndim-1];
                    size_t right_idx = r_offset + k*rstrides[max_ndim-2] + j*rstrides[max_ndim-1];
                    sum += left->data[left_idx] * right->data[right_idx];
                }
                output->data[b*M*N + i*N + j] = sum;
            }
        }
    }
    free(lshape);
    free(rshape);
    free(out_shape);
    free(lstrides);
    free(rstrides);
}

void reduce_sum(const Array *input, Array *output, const int *axes, int num_axes) 
{
    // Initialize output to zero
    for (size_t i = 0; i < output->size; i++) {
        output->data[i] = 0.0f;
    }
    // Create reduction mask (1=reduce, 0=keep)
    int *reduce_mask = (int *)calloc(input->ndim, sizeof(int));
    if (axes == NULL) {
        // Reduce all dimensions when axes is NULL
        for (int d = 0; d < input->ndim; d++) {
            reduce_mask[d] = 1;
        }
    } else {
        // Mark specified axes for reduction
        for (int i = 0; i < num_axes; i++) {
            if (axes[i] >= 0 && axes[i] < input->ndim) {
                reduce_mask[axes[i]] = 1;
            }
        }
    }

    // Iterate through input and accumulate sums
    for (size_t i = 0; i < input->size; i++) {
        // Compute output index
        size_t out_idx = 0;
        int temp = i;
        int out_stride = 1;
        
        for (int d = input->ndim - 1; d >= 0; d--) {
            int dim_size = input->shape[d];
            int idx = temp % dim_size;
            temp /= dim_size;
            if (!reduce_mask[d]) {
                out_idx += idx * out_stride;
                out_stride *= output->shape[d];
            }
        }
        output->data[out_idx] += input->data[i];
    }

    free(reduce_mask);
}


void relu(Array *output, const Array *input) 
{
    const float *input_data = input->data;
    float *output_data = output->data;

    for (size_t i = 0; i < output->size; i++) {
        output_data[i] = MAX(0.0f, input_data[i]);
    }

}

void squared_error(Array *output, Array *input, Array *target) 
{
    binary_array_iterator(input, target, output, squared_diff);
}

void reduce_mean(Array *output, Array *input, Array *axes, void *keepdim) 
{
    int num_axes = 0;
    if (axes != NULL) {
        num_axes = axes->ndim;
    }
    else {
        num_axes = input->ndim;
    }
    // TODO: keepdim is NULL, add keepdim support.
    size_t N = 1;
    if (axes == NULL) {
        N = input->size;
    } else {
        for (int i=0; i<axes->ndim; i++) {
            N *= input->shape[(int) axes->data[i]];
        }
    }
    // Convert float axes to int
    int *int_axes = NULL;
    if (axes != NULL) {
        int_axes = (int *)malloc(num_axes * sizeof(int));
        for (int i = 0; i < num_axes; i++) {
            int_axes[i] = (int)(axes->data[i]);
        }
    }
    reduce_sum(input, output, int_axes, num_axes);
    for (size_t i=0; i<output->size; i++) {
        output->data[i] /= N;
    }
}


void add_grad(Array *gradient, int idx, Array *output, Array *left, Array *right, Array *leftGradient, Array *rightGradient) 
{
    if (idx == 0) {
        // Determine broadcasted dimensions for left
        int ndim_diff = gradient->ndim - left->ndim;
        if (ndim_diff == 0 )
        {
            add(leftGradient, gradient, leftGradient);
        }
        else{
            int *reduce_axes = (int *)malloc(ndim_diff * sizeof(int));
            for (int i = 0; i < ndim_diff; i++) {
                reduce_axes[i] = i;
            }
            reduce_sum(gradient, leftGradient, reduce_axes, ndim_diff);
            free(reduce_axes);
        }
    } else {
        // Determine broadcasted dimensions for right
        int ndim_diff = gradient->ndim - right->ndim;
        if (ndim_diff == 0 )
        {
            add(rightGradient, gradient, rightGradient);
        }
        else{
            int *reduce_axes = (int *)malloc(ndim_diff * sizeof(int));
            for (int i = 0; i < ndim_diff; i++) {
                reduce_axes[i] = i;
            }
            reduce_sum(gradient, rightGradient, reduce_axes, ndim_diff);
            free(reduce_axes);
        }
    }
}

void multiplication_grad(Array *gradient, int idx, Array *output, Array *left, Array *right, Array *leftGradient, Array *rightGradient)
{
    Array *temp;
    if (idx == 0) {
        temp = create_full_struct(0.0f, gradient->ndim, gradient->shape);
        // Determine broadcasted dimensions for left
        int ndim_diff = gradient->ndim - left->ndim;
        if (ndim_diff == 0)
        {
            multiplication(temp, gradient, right);
            add(leftGradient, temp, leftGradient);
        }
        else{
            binary_array_iterator(gradient, right, temp, multiply_lambda);
            int *reduce_axes = (int *)malloc(ndim_diff * sizeof(int));
            for (int i = 0; i < ndim_diff; i++) 
                reduce_axes[i] = i;
            reduce_sum(temp, leftGradient, reduce_axes, ndim_diff);
            free(reduce_axes);
            //add(leftGradient, temp, leftGradient);
        }
    } else {
        temp = create_full_struct(0.0f, gradient->ndim, gradient->shape);
        int ndim_diff = gradient->ndim - right->ndim;
        if (ndim_diff == 0)
        {
            multiplication(temp, gradient, left);
            add(rightGradient, temp, rightGradient);
        }
        else{
            binary_array_iterator(gradient, left, temp, multiply_lambda);
            // Determine broadcasted dimensions for right
            int *reduce_axes = malloc(ndim_diff * sizeof(int));
            for (int i = 0; i < ndim_diff; i++) 
                reduce_axes[i] = i;
            reduce_sum(temp, rightGradient, reduce_axes, ndim_diff);
            free(reduce_axes);
        }
    }
    free(temp->data);
    free(temp->shape);
    free(temp->strides);
    free(temp);
}

void transpose_grad(const Array *gradient, int idx, Array *output, const Array *left, const Array *right, Array *leftGradient, void *axes)
{
    if (axes == NULL) {
        // For the standard 2D transpose, the gradient is just the transpose.
        transpose(leftGradient, gradient, axes);
    } else {
        int ndim = gradient->ndim;
        int *axes_int = (int *)axes;
        int *inv_axes = (int *)malloc(ndim * sizeof(int));
        invert_permutation(axes_int, inv_axes, ndim);
        // Use the inverse permutation in the transpose.
        transpose(leftGradient, gradient, (void *)inv_axes);
        free(inv_axes);
    }
}

void relu_grad(const Array *outputGradient, int idx, Array *output, Array *input, Array *inputGradient) 
{
    // Compute the broadcasted strides for input relative to output_grad's shape
    int *input_b_strides = broadcastStride(input, outputGradient->shape, outputGradient->ndim);
    const float *input_data = input->data;
    const float *output_grad_data = outputGradient->data;
    float *input_grad_data = inputGradient->data;

    // Iterate over each element in the output gradient
    for (size_t i = 0; i < outputGradient->size; i++) {
        // Find the corresponding index in the input array
        size_t input_idx = loc(i, outputGradient->shape, input_b_strides, outputGradient->ndim);

        // If input element was positive, accumulate the gradient
        if (input_data[input_idx] > 0.0f) {
            input_grad_data[input_idx] += output_grad_data[i];
        }
    }

    free(input_b_strides);
}

void squared_error_grad(Array *outputGradient, int idx, Array *output, Array *input, Array *target, Array *inputGradient, Array *targetGradient) 
{
    float sign = (idx == 0) ? 1.0f : -1.0f;
    float factor = 2.0f * sign;

    // Compute broadcasted strides for input and target relative to output_grad's shape
    int *input_b_strides = broadcastStride(input, outputGradient->shape, outputGradient->ndim);
    int *target_b_strides = broadcastStride(target, outputGradient->shape, outputGradient->ndim);

    // Validate allocations
    if (!input_b_strides || !target_b_strides) {
        fprintf(stderr, "Memory allocation failed in squared_error_grad\n");
        exit(EXIT_FAILURE);
    }

    const float *input_data = input->data;
    const float *target_data = target->data;
    const float *grad_data_in = outputGradient->data;

    // Determine which gradient array to update
    Array *grad_array = (idx == 0) ? inputGradient : targetGradient;
    float *grad_data_out = grad_array->data;

    // Iterate over each element in the output gradient
    for (size_t i = 0; i < outputGradient->size; ++i) {
        // Get broadcasted indices for input and target
        size_t input_idx = loc(i, outputGradient->shape, input_b_strides, outputGradient->ndim);
        size_t target_idx = loc(i, outputGradient->shape, target_b_strides, outputGradient->ndim);

        // Compute gradient contribution
        float diff = input_data[input_idx] - target_data[target_idx];
        float grad = factor * diff * grad_data_in[i];

        // Accumulate gradient into the appropriate array
        grad_data_out[input_idx] += grad;
    }

    // Cleanup
    free(input_b_strides);
    free(target_b_strides);
}

void reduce_mean_grad(Array *outputGradient, int idx, Array *output, Array *input, Array *axes, bool keepdim, Array *inputGradient, Array *num_axes, bool keepdimGradient) 
{
    // Target is not differentiable
    if (idx != 0) 
        return;
    size_t N;
    if (axes == NULL) {
        N = input->size;
    } else {
        N = 1;
        for (int i=0; i<num_axes; i++) {
            N *= input->shape[(int) axes->data[i]];
        }
    }

    int *bcast_strides = broadcastStride(outputGradient, input->shape, input->ndim);
    for (size_t i=0; i<input->size; i++) {
        size_t grad_idx = loc(i, input->shape, bcast_strides, input->ndim);
        inputGradient->data[i] += outputGradient->data[grad_idx] / N;
    }
    free(bcast_strides);
}
void matrix_multiplication_grad(const Array *gradient, int idx, Array *output ,const Array *left, const Array *right, Array *leftGradient, Array *rightGradient) 
{
    // Get batch dimensions.
    int left_batch_ndim = left->ndim - 2;
    int right_batch_ndim = right->ndim - 2;
    int gradient_batch_ndim = gradient->ndim - 2;

    // Matrix dimensions.
    int M = left->shape[left->ndim - 2];
    int K = left->shape[left->ndim - 1];
    int K2 = right->shape[right->ndim - 2];
    int N = right->shape[right->ndim - 1];

    if (K != K2) {
        fprintf(stderr, "Dimension mismatch in matmul backward: A's K (%d) != B's K (%d)\n", K, K2);
        exit(EXIT_FAILURE);
    }

    // Zero out gradients leftGradient and rightGradient.
    if (idx == 0) {
        for (size_t i = 0; i < leftGradient->size; i++) {
            leftGradient->data[i] = 0.0f;
        }
    } else {
        for (size_t i = 0; i < rightGradient->size; i++) {
            rightGradient->data[i] = 0.0f;
        }
    }

    // Total number of broadcasted batches in gradient.
    int batch_size = prod(gradient->shape, gradient_batch_ndim);

    #pragma omp parallel for
    // Iterate over each broadcasted batch.
    for (int b = 0; b < batch_size; b++) {
        size_t left_base_offset = loc(b, left->shape, left->strides, left_batch_ndim);
        size_t right_base_offset = loc(b, right->shape, right->strides, right_batch_ndim);
        size_t gradient_base_offset = loc(b, gradient->shape, gradient->strides, gradient_batch_ndim);

        if (idx == 0) {
            // --- Compute gradient for Left ---
            // leftGradient[i, k] += sum_j gradient[i, j] * right[k, j]
            for (int i = 0; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    float grad = 0.0f;
                    for (int j = 0; j < N; j++) {
                        size_t gradient_idx = gradient_base_offset + i * gradient->strides[gradient->ndim - 2] + j * gradient->strides[gradient->ndim - 1];
                        size_t right_idx = right_base_offset + k * right->strides[right->ndim - 2] + j * right->strides[right->ndim - 1];
                        grad += gradient->data[gradient_idx] * right->data[right_idx];
                    }
                    size_t leftGradient_idx = left_base_offset + i * left->strides[left->ndim - 2] + k * left->strides[left->ndim - 1];
                    leftGradient->data[leftGradient_idx] += grad;
                }
            }
        } else {
            // --- Compute gradient for Right ---
            // dB[k, j] += sum_i left[i, k] * gradient[i, j]
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < K; k++) {
                    float grad = 0.0f;
                    for (int i = 0; i < M; i++) {
                        size_t gradient_idx = gradient_base_offset +
                                        i * gradient->strides[gradient->ndim - 2] +
                                        j * gradient->strides[gradient->ndim - 1];
                        size_t A_idx = left_base_offset +
                                    i * left->strides[left->ndim - 2] +
                                    k * left->strides[left->ndim - 1];
                        grad += left->data[A_idx] * gradient->data[gradient_idx];
                    }
                    // Here, we treat B as if its shape is [N, K]
                    size_t rightGradient_idx = right_base_offset +
                                    k * right->strides[right->ndim - 2] +
                                    j * right->strides[right->ndim - 1];
                    rightGradient->data[rightGradient_idx] += grad;
                }
            }
        }
    }
}