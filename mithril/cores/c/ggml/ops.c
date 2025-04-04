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



struct ggml_tensor * add(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * res = ggml_add(ctx, left, right);
    return res;

}

struct ggml_tensor * multiplication(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * res = ggml_mul(ctx, left, right);
    return res;
}

struct ggml_tensor * scalar_multiply(struct ggml_context * ctx, struct ggml_tensor * left, float scale) {
    struct ggml_tensor * res = ggml_scale(ctx, left, scale);
    return res;
}

struct ggml_tensor * subtract(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * res = ggml_sub(ctx, left, right);
    return res;
}

struct ggml_tensor * broadcast_to(struct ggml_context * ctx, struct ggml_tensor * input, int dim1, int dim2, int dim3, int dim4) {
    // If the target shape is not 4D, the empty dimensions are need to be set to 1
    struct ggml_tensor * shape = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 0, 0, 0, 0);
    shape->ne[0] = dim1;
    shape->ne[1] = dim2;
    shape->ne[2] = dim3;
    shape->ne[3] = dim4;
    struct ggml_tensor * res = ggml_repeat(ctx, input, shape);
    return res; 

}

struct ggml_tensor * transpose(struct ggml_context * ctx, struct ggml_tensor * input, struct ggml_tensor * axes) {
    struct ggml_tensor * res =  ggml_cont(ctx,ggml_transpose(ctx, ggml_dup(ctx,input)));
    return res;
}

struct ggml_tensor * matrix_multiplication(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) {

    struct ggml_tensor * res = ggml_mul_mat(ctx,transpose(ctx,right,right),left );
    return res;
}

struct ggml_tensor * relu(struct ggml_context * ctx, struct ggml_tensor * input) {
    
    struct ggml_tensor * res = ggml_relu(ctx, input);
    return res;
}

struct ggml_tensor * squared_error(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * res = ggml_sub(ctx, left, right);
    res = ggml_sqr(ctx, res);
    return res;
}

struct ggml_tensor * reduce_mean(struct ggml_context * ctx, struct ggml_tensor * input, struct ggml_tensor * axes,struct ggml_tensor * keepdim) {
    // TODO: keepdim and axes in NULL and they are not used. 
    // Add keepdim and axes support.
    int64_t num_elements = ggml_nelements(input); 
    struct ggml_tensor * sum = ggml_sum(ctx, input);

    struct ggml_tensor * mean = ggml_scale(ctx, sum, 1.0f / num_elements);
    return mean;
}

struct ggml_tensor * add_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left, struct ggml_tensor * right)
{ 
    return gradient;
}

struct ggml_tensor * multiplication_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left, struct ggml_tensor * right)
{
    if (idx == 0){
        return multiplication(ctx, gradient, right);
    }
    else{
        return multiplication(ctx, gradient, left);
    }
}

struct ggml_tensor * transpose_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left,  struct ggml_tensor * right) {     
    return transpose(ctx,gradient,gradient);
}

struct ggml_tensor * matrix_multiplication_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left,  struct ggml_tensor * right) {
    struct ggml_tensor * grad;
    if (idx == 0) {
        grad = ggml_mul_mat(ctx, right , gradient);
        return grad;
    } else if (idx == 1) {
        grad = ggml_mul_mat(ctx,transpose(ctx,gradient, gradient),transpose(ctx,left, left));
       
        return grad;
    } else {
        GGML_ASSERT(false && "Invalid index for matrix multiplication gradient.");
        return NULL;
    }
}

struct ggml_tensor * relu_grad(struct ggml_context * ctx, struct ggml_tensor * output_grad, int idx, struct ggml_tensor * output, struct ggml_tensor * input ) {
    struct ggml_tensor * mask = ggml_step(ctx, input); // Step function: 1 if input > 0, else 0
    struct ggml_tensor * grad = ggml_mul(ctx, mask,output_grad);
    return grad;
}

struct ggml_tensor * squared_error_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * scaled_diff = ggml_scale(ctx, ggml_sub(ctx, left, right), 2.0f);
    struct ggml_tensor * grad = ggml_mul(ctx, gradient, scaled_diff);

    if (idx == 1) {
        grad = ggml_neg(ctx, grad);
    }
    return grad;
}

struct ggml_tensor * reduce_mean_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * left, struct ggml_tensor * right, struct ggml_tensor * axes,struct ggml_tensor * keepdim) {
    struct ggml_tensor * input = right;
    const int64_t size = ggml_nelements(input);

    struct ggml_tensor * scaled_grad = ggml_scale(ctx, gradient, 1.0f / (float)size);
    struct ggml_tensor * grad = ggml_repeat(ctx, scaled_grad, input);
    return grad;
}