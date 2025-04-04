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

#ifndef OPS_H
#define OPS_H
#include "array.h"
#include <assert.h>
#include <stdbool.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void add(Array *output, Array *left, Array *right);
void multiplication(Array *output, Array *left, Array *right);
void subtract(Array *output, Array *left, Array *right);
void matrix_multiplication(Array *output, const Array *left, const Array *right);
void transpose(Array *output,const Array *input, void * axes);
void transpose_grad(const Array *gradient, int idx, Array *output , const Array *left, const Array *right, Array *leftGradient, void * axes);
void matrix_multiplication_grad(const Array *gradient, int idx, Array *output ,const Array *left, const Array *right, Array *leftGradient, Array *rightGradient);
void add_grad(Array *gradient, int idx, Array *output, Array *left, Array *right, Array *leftGradient, Array *rightGradient);
void multiplication_grad(Array *gradient, int idx, Array *output, Array *left, Array *right, Array *leftGradient, Array *rightGradient);
void reduce_sum(const Array *input, Array *output, const int *axes, int num_axes);
void relu(Array *output, const Array *input);
void relu_grad(const Array *outputGradient, int idx, Array *output, Array *input, Array *inputGradient);
void squared_error(Array *output, Array *input, Array *target) ;
void squared_error_grad(Array *outputGradient, int idx, Array *output, Array *input, Array *target, Array *inputGradient, Array *targetGradient);
void reduce_mean(Array *output, Array *input, Array *axes, void *keepdim);
void reduce_mean_grad(Array *outputGradient, int idx, Array *output, Array *input, Array *axes, bool keepdim, Array *inputGradient, int num_axes, bool keepdimGradient);

#endif