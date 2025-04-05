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

#define SWAP(a, b) do { \
    typeof(a) temp = a; \
    a = b;              \
    b = temp;           \
} while (0)

typedef void (*Op)(float *output, float input);

int *broadcastStride(const Array *t1, const int *shape, const int ndim);
size_t loc(size_t idx, const int *shapes, const int *strides, const int ndim);
void binary_array_iterator(const Array *left, const Array *right, Array *out, float (*op)(float, float));
void reduce_contiguous_all(const Array *input, Array *out, float init_val, Op op);
void reduce_contiguous_dim(const float *input_data, float *output_data, const int *reduction_size, const int *reduction_strides, size_t offset, size_t dim, size_t max_dim, Op op);
void reduce_contiguous(const Array *input, Array *out, const int *axes, size_t num_axes, float init_val, Op op);
int* pad_shape(const Array *arr, int target_ndim) ;
int* compute_strides(const int *shape, int ndim);
int prod(const int *arr, int len);
void invert_permutation(const int *axes, int *inv_axes, int ndim);
void scalar_add(Array *output, Array *input, float scalar);
void scalar_multiply(Array *output, Array *input, float scalar);
void scalar_subtract(Array *output, Array *input, float scalar);

#endif