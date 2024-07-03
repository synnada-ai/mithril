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

#ifndef UTILS_H
#define UTILS_H
#include "array.h"

typedef void (*Op)(float *output, float input);

/* Creates new strides for broadcasting */
int *broadcastStride(const Array *t1, const int *shape, const int ndim)
{
    int diff = ndim - t1->ndim;
    int *oldStrides = t1->strides;
    int *newStrides = (int *)malloc(ndim * sizeof(int));

    for (size_t i = 0; i < ndim; i++)
        newStrides[i] = 0;

    for (size_t i = diff; i < t1->ndim; i++)
    {
        if (shape[i] == 1 || t1->shape[i - diff] == 1)
            newStrides[i] = 0;
        else
            newStrides[i] = oldStrides[i - diff];
    }

    return newStrides;
}

/* Get the real index of the given index wrt strides and shapes */
size_t loc(size_t idx, const int *shapes, const int *strides, const int ndim)
{
    size_t loc = 0;
    for (size_t i = ndim - 1; i >= 0 && idx > 0; i--)
    {
        loc += (idx % shapes[i]) * strides[i];
        idx /= shapes[i];
    }
    return loc;
}

/* Handles binary operations on two arrays */
void binary_array_iterator(const Array *left, const Array *right, Array *out, float (*op)(float, float))
{
    const float *left_data = left->data;
    const float *right_data = right->data;

    int *left_b_strides = broadcastStride(left, out->shape, out->ndim);
    int *right_b_strides = broadcastStride(right, out->shape, out->ndim);

    for (size_t i = 0; i < out->size; i++)
    {
        // TODO: Use loc only when the Tensor is not contiguous
        size_t left_idx = loc(i, out->shape, left_b_strides, out->ndim);
        size_t right_idx = loc(i, out->shape, right_b_strides, out->ndim);
        out->data[i] = op(left_data[left_idx], right_data[right_idx]);
    }

    free(left_b_strides);
    free(right_b_strides);
}





/* If the input Array  is contiguous, and the reduction type is all, reduce the array directly */
void reduce_contiguous_all(const Array *input, Array *out, float init_val, Op op)
{
    const float *input_data = input->data;
    float *output_data = out->data;
    *output_data = init_val;

    for (size_t i = 0; i < input->size; i++)
    {
        op(output_data, input_data[i]);
    }
}

void reduce_contiguous_dim(const float *input_data, float *output_data, const int *reduction_size, const int *reduction_strides, size_t offset, size_t dim, size_t max_dim, Op op)
{
    if (dim == max_dim - 1)
    {
        for (size_t i = 0; i < reduction_size[dim]; i++)
        {
            op(output_data, input_data[offset + i * reduction_strides[dim]]);
        }
    }
    else
    {
        for (size_t i = 0; i < reduction_size[dim]; i++)
        {
            reduce_contiguous_dim(input_data, output_data, reduction_size, reduction_strides, offset + i * reduction_strides[dim], dim + 1, max_dim, op);
        }
    }
}

/* If the input Array is contiguous, and reduction only some of the axes will be reduced */
void reduce_contiguous(const Array *input, Array *out, const int *axes, size_t num_axes, float init_val, Op op)
{
    const int *in_shapes = input->shape;
    const int *in_strides = input->strides;

    int *reduction_size = (int *)malloc(num_axes * sizeof(int));
    int *reduction_strides = (int *)malloc(num_axes * sizeof(int));

    reduction_size[0] = in_shapes[axes[0]];
    reduction_strides[0] = in_strides[axes[0]];

    float *output_data = out->data;

    for (size_t i = 1; i < num_axes; i++)
    {
        if (axes[i] - 1 == axes[i - 1])
        {
            reduction_size[i - 1] *= in_shapes[axes[i]];
            reduction_strides[i - 1] = in_strides[axes[i]];
        }
        else
        {
            reduction_size[i] = in_shapes[axes[i]];
            reduction_strides[i] = in_strides[axes[i]];
        }
    }

    for (size_t i = 0; i < out->size; i++, output_data++)
    {
        *output_data = init_val;
        reduce_contiguous_dim(input->data, output_data, reduction_size, reduction_strides, i, 0, num_axes, op);
    }

    free(reduction_size);
    free(reduction_strides);
}

#endif