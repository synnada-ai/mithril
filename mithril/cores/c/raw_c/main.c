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

#include <stdio.h>
#include "cbackend.h"


void evaluate(Array * left, Array * left2, Array * output, Array * output2, Array * right) {
    add(output, left, right);
    add(output2, left2, output);
}

void evaluate_gradients(Array * left, Array * left2, Array * left2_grad, Array * left_grad, Array * output, Array * output2, Array * output2_grad, Array * output_grad, Array * right, Array * right_grad) {
    add_backward(output2_grad, 0, output2, left2, output, left2_grad, output_grad);
    add_backward(output2_grad, 1, output2, left2, output, left2_grad, output_grad);
    add_backward(output_grad, 0, output, left, right, left_grad, right_grad);
    add_backward(output_grad, 1, output, left, right, left_grad, right_grad);
}

int main()
{
    int left_shape[] = {5, 5};
    int right_shape[] = {5, 5};
    int out_shape[] = {5, 5};
    Array *left = create_full_struct(1, 2, left_shape);
    Array *left_grad = create_full_struct(0, 2, left_shape);
    Array *left2 = create_full_struct(1, 2, left_shape);
    Array *left2_grad = create_full_struct(0, 2, left_shape);
    Array *right = create_full_struct(4, 2, right_shape);
    Array *right_grad = create_full_struct(0, 2, right_shape);
    Array *out = create_empty_struct(2, out_shape);
    Array *out_grad = create_full_struct(0.33, 2, out_shape);
    Array *out2 = create_empty_struct(2, out_shape);
    Array *out2_grad = create_full_struct(0.33, 2, out_shape);

    evaluate_gradients(left, left2, left2_grad, left_grad, out, out2, out2_grad, out_grad, right,  right_grad);
    printf("Left grad:\n");
    
}



