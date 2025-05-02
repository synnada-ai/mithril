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

typedef struct ggml_tensor g_tensor;
typedef struct ggml_context g_context;

g_tensor *add(g_context *ctx, g_tensor *left, g_tensor *right);
g_tensor *multiplication(g_context *ctx, g_tensor *left, g_tensor *right);
g_tensor *scalar_multiply(g_context *ctx, g_tensor *left, float scale);
g_tensor *subtract(g_context *ctx, g_tensor *left, g_tensor *right);
g_tensor *transpose(g_context *ctx, g_tensor *input, g_tensor *axes);
g_tensor *matrix_multiplication(g_context *ctx, g_tensor *left,
                                g_tensor *right);
g_tensor *relu(g_context *ctx, g_tensor *input);
g_tensor *squared_error(g_context *ctx, g_tensor *left, g_tensor *right);
g_tensor *reduce_mean(g_context *ctx, g_tensor *input, g_tensor *axes,
                      g_tensor *keepdim);
g_tensor *broadcast_to(g_context *ctx, g_tensor *input, int dim1, int dim2,
                       int dim3, int dim4);

g_tensor *add_grad(g_context *ctx, g_tensor *gradient, int idx,
                   g_tensor *output, g_tensor *left, g_tensor *right);
g_tensor *multiplication_grad(g_context *ctx, g_tensor *gradient, int idx,
                              g_tensor *output, g_tensor *left,
                              g_tensor *right);
g_tensor *transpose_grad(g_context *ctx, g_tensor *gradient, int idx,
                         g_tensor *output, g_tensor *left, g_tensor *right);
g_tensor *matrix_multiplication_grad(g_context *ctx, g_tensor *gradient,
                                     int idx, g_tensor *output, g_tensor *left,
                                     g_tensor *right);
g_tensor *relu_grad(g_context *ctx, g_tensor *output_grad, int idx,
                    g_tensor *output, g_tensor *input);
g_tensor *squared_error_grad(g_context *ctx, g_tensor *gradient, int idx,
                             g_tensor *output, g_tensor *left, g_tensor *right);
g_tensor *reduce_mean_grad(g_context *ctx, g_tensor *gradient, int idx,
                           g_tensor *left, g_tensor *right, g_tensor *axes,
                           g_tensor *keepdim);

#endif
