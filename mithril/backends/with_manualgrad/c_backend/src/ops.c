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

float add_lambda(float x, float y)
{
    return x + y;
}

float multiply_lambda(float x, float y)
{
    return x * y;
}

void add(Array *output, Array *left, Array *right)
{
    binary_array_iterator(left, right, output, add_lambda);
}

void multiplication(Array *output, Array *left, Array *right)
{
    binary_array_iterator(left, right, output, multiply_lambda);
}


void add_grad(Array *gradient, int idx, Array *output, Array *left, Array *right, Array *leftGradient, Array *rightGradient)
{
    if (idx == 0)
        add(leftGradient, gradient, leftGradient);
    else
        add(rightGradient, gradient, rightGradient);
}


void multiplication_grad(Array *gradient, int idx, Array *output, Array *left, Array *right, Array *leftGradient, Array *rightGradient)
{
    Array *temp;
    if (idx == 0){
        temp = create_full_struct(0.0f, leftGradient->ndim, leftGradient->shape);
        multiplication(temp, gradient, right);
        add(leftGradient, temp, leftGradient);
    }
    else{
        temp = create_full_struct(0.0f, rightGradient->ndim, rightGradient->shape);
        multiplication(temp, gradient, left);
        add(rightGradient, temp, rightGradient);
    }

    free(temp->data);
    free(temp->shape);
    free(temp->strides);
    free(temp);
}
