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

#include "ops.h"
#include <stdio.h>

g_tensor * add(g_context * ctx, g_tensor * left, g_tensor * right) {
    return ggml_add(ctx, left, right);
}

g_tensor * multiplication(g_context * ctx, g_tensor * left, g_tensor * right) {
    return ggml_mul(ctx, left, right);
}

g_tensor * scalar_multiply(g_context * ctx, g_tensor * left, float scale) {
    return ggml_scale(ctx, left, scale);
}

g_tensor * subtract(g_context * ctx, g_tensor * left, g_tensor * right) {
    return ggml_sub(ctx, left, right);
}

g_tensor * broadcast_to(g_context * ctx, g_tensor * input, int dim1, int dim2, int dim3, int dim4) {
    g_tensor * shape = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 0, 0, 0, 0);
    shape->ne[0] = dim1;
    shape->ne[1] = dim2;
    shape->ne[2] = dim3;
    shape->ne[3] = dim4;
    return ggml_repeat(ctx, input, shape);
}

g_tensor * transpose(g_context * ctx, g_tensor * input, g_tensor * axes) {
    return ggml_cont(ctx, ggml_transpose(ctx, ggml_dup(ctx, input)));
}

g_tensor * matrix_multiplication(g_context * ctx, g_tensor * left, g_tensor * right) {
    return ggml_mul_mat(ctx, transpose(ctx, right, right), left);
}

g_tensor * relu(g_context * ctx, g_tensor * input) {
    return ggml_relu(ctx, input);
}

g_tensor * squared_error(g_context * ctx, g_tensor * left, g_tensor * right) {
    return ggml_sqr(ctx, ggml_sub(ctx, left, right));
}

g_tensor * reduce_mean(g_context * ctx, g_tensor * input, g_tensor * axes, g_tensor * keepdim) {
    return ggml_scale(ctx, ggml_sum(ctx, input), 1.0f / ggml_nelements(input));
}

g_tensor * add_grad(g_context * ctx, g_tensor * gradient, int idx, g_tensor * output, g_tensor * left, g_tensor * right) { 
    return gradient;
}

g_tensor * multiplication_grad(g_context * ctx, g_tensor * gradient, int idx, g_tensor * output, g_tensor * left, g_tensor * right) {
    if (idx == 0) {
        return multiplication(ctx, gradient, right);
    } else {
        return multiplication(ctx, gradient, left);
    }
}

g_tensor * transpose_grad(g_context * ctx, g_tensor * gradient, int idx, g_tensor * output, g_tensor * left, g_tensor * right) {     
    return transpose(ctx, gradient, gradient);
}

g_tensor * matrix_multiplication_grad(g_context * ctx, g_tensor * gradient, int idx, g_tensor * output, g_tensor * left, g_tensor * right) {
    if (idx == 0) {
        return ggml_mul_mat(ctx, right, gradient);
    } else if (idx == 1) {
        return ggml_mul_mat(ctx, transpose(ctx, gradient, gradient), transpose(ctx, left, left));
    } else {
        GGML_ASSERT(false && "Invalid index for matrix multiplication gradient.");
        return NULL;
    }
}

g_tensor * relu_grad(g_context * ctx, g_tensor * output_grad, int idx, g_tensor * output, g_tensor * input) {
    return ggml_mul(ctx, ggml_step(ctx, input), output_grad);
}

struct ggml_tensor * squared_error_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * scaled_diff = ggml_scale(ctx, ggml_sub(ctx, left, right), 2.0f);
    struct ggml_tensor * grad = ggml_mul(ctx, gradient, scaled_diff);

    if (idx == 1) {
        grad = ggml_neg(ctx, grad);
    }
    return grad;
    return (idx == 1) ? ggml_neg(ctx, grad) : grad;
}

struct ggml_tensor * reduce_mean_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * left, struct ggml_tensor * right, struct ggml_tensor * axes,struct ggml_tensor * keepdim) {
    struct ggml_tensor * input = right;
    const int64_t size = ggml_nelements(input);

    struct ggml_tensor * scaled_grad = ggml_scale(ctx, gradient, 1.0f / (float)size);
    return ggml_repeat(ctx, scaled_grad, input);
}