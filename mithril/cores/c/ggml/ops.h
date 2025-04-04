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

#ifndef MITHRIL_GGML_OPS_H
#define MITHRIL_GGML_OPS_H

#include "ggml/include/ggml.h"
#include "utils.h"

struct ggml_tensor * add(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right);
struct ggml_tensor * multiplication(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right);
struct ggml_tensor * scalar_multiply(struct ggml_context * ctx, struct ggml_tensor * left, float scale);
struct ggml_tensor * subtract(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right);
struct ggml_tensor * transpose(struct ggml_context * ctx, struct ggml_tensor * input, struct ggml_tensor * axes);
struct ggml_tensor * matrix_multiplication(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right);
struct ggml_tensor * relu(struct ggml_context * ctx, struct ggml_tensor * input);
struct ggml_tensor * squared_error(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) ;
struct ggml_tensor * reduce_mean(struct ggml_context * ctx, struct ggml_tensor * input, struct ggml_tensor * axes,struct ggml_tensor * keepdim);
struct ggml_tensor * broadcast_to(struct ggml_context * ctx, struct ggml_tensor * input, int dim1, int dim2, int dim3, int dim4);


struct ggml_tensor * add_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left, struct ggml_tensor * right);
struct ggml_tensor * multiplication_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left, struct ggml_tensor * right);
struct ggml_tensor * transpose_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left,  struct ggml_tensor * right);
struct ggml_tensor * matrix_multiplication_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left,  struct ggml_tensor * right);
struct ggml_tensor * relu_grad(struct ggml_context * ctx, struct ggml_tensor * output_grad, int idx, struct ggml_tensor * output, struct ggml_tensor * input );
struct ggml_tensor * squared_error_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left, struct ggml_tensor * right);
struct ggml_tensor * reduce_mean_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * left, struct ggml_tensor * right, struct ggml_tensor * axes,struct ggml_tensor * keepdim);

#endif
