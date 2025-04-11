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

#ifndef MITHRIL_GGML_UTILS_H
#define MITHRIL_GGML_UTILS_H

#include "ggml/include/ggml.h"
#include "ops.h"


int accumulate_grads_helper(struct ggml_tensor * gradient, struct ggml_tensor * input);
struct ggml_tensor * accumulate_grads(struct ggml_context * ctx, struct ggml_tensor * gradient, struct ggml_tensor * input);


struct ggml_tensor * accumulate_grads(struct ggml_context * ctx, struct ggml_tensor * gradient, struct ggml_tensor * input)
{
    struct ggml_tensor * res;
    if(ggml_n_dims(gradient) == ggml_n_dims(input)){
        return gradient;
    }
    int axes = accumulate_grads_helper(gradient, input);
    if(axes == 0)
    {
        if(ggml_n_dims(gradient) == 3)
        {
            struct ggml_tensor * temp = ggml_sum_rows(ctx,ggml_cont(ctx,ggml_permute(ctx, gradient,2,1,0,3)));
            res = ggml_permute(ctx, temp, 2,1,0,3);
        }
        else
        {
            res = ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, ggml_cont(ctx,ggml_transpose(ctx, gradient)))));
        }
        
    }
    else if(axes == 1){
        struct ggml_tensor * temp = ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, ggml_cont(ctx,ggml_transpose(ctx, gradient)))));
        struct ggml_tensor * temp1  = ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, ggml_cont(ctx,ggml_transpose(ctx, temp)))));
        struct ggml_tensor * temp2 = ggml_sum_rows(ctx,ggml_cont(ctx,ggml_permute(ctx, temp1,2,1,0,3)));
        res = ggml_permute(ctx, temp2, 2,1,0,3);
    }
    else
    {
        res = ggml_sum_rows(ctx,gradient);
    }
    return res;
}

int accumulate_grads_helper(struct ggml_tensor * gradient, struct ggml_tensor * input)
{
    int axes;
    int diff = ggml_n_dims(gradient) - ggml_n_dims(input);

    for(int i = 0; i < diff; i++)
    {
        axes = i;
    }

    for(int i = 0; i < ggml_n_dims(input); i++)
    {
        if(input->ne[i] != gradient->ne[i] & gradient->ne[i] != 1)
        {
            axes += ggml_n_dims(gradient) - i + 1;
        } 
    }

    return axes;
}

#endif