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

#include "../../mithril/cores/c/raw_c/cbackend.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define FLOAT_TOLERANCE 1e-6

void assert_array_equal(const Array *result, const float *expected, int size) {
  assert(result->size == size);
  for (int i = 0; i < size; i++) {
    if (fabs(result->data[i] - expected[i]) > FLOAT_TOLERANCE) {
      printf("Mismatch at index %d: %f vs %f\n", i, result->data[i],
             expected[i]);
      exit(1);
    }
  }
}

void test_add_same_shape() {
  printf("test_add_same_shape\n");
  int shape[] = {2, 2};
  float left_data[] = {1.0, 2.0, 3.0, 4.0};
  float right_data[] = {5.0, 6.0, 7.0, 8.0};
  float expected[] = {6.0, 8.0, 10.0, 12.0};

  Array *left = create_struct(left_data, 2, shape);
  Array *right = create_struct(right_data, 2, shape);
  Array *output = create_empty_struct(2, shape);

  add(output, left, right);
  assert_array_equal(output, expected, 4);

  delete_struct(left);
  delete_struct(right);
  delete_struct(output);
}

void test_add_broadcast() {
  printf("test_add_broadcast\n");
  int left_shape[] = {2, 3};
  float left_data[] = {1, 2, 3, 4, 5, 6};
  int right_shape[] = {1, 3};
  float right_data[] = {10, 20, 30};
  float expected[] = {11, 22, 33, 14, 25, 36};

  Array *left = create_struct(left_data, 2, left_shape);
  Array *right = create_struct(right_data, 2, right_shape);
  Array *output = create_empty_struct(2, left_shape);

  add(output, left, right);
  assert_array_equal(output, expected, 6);

  delete_struct(left);
  delete_struct(right);
  delete_struct(output);
}

void test_matrix_multiplication_2d() {
  printf("test_matrix_multiplication_2d\n");
  int left_shape[] = {2, 2};
  float left_data[] = {1, 2, 3, 4};
  int right_shape[] = {2, 2};
  float right_data[] = {5, 6, 7, 8};
  float expected[] = {19, 22, 43, 50};

  Array *left = create_struct(left_data, 2, left_shape);
  Array *right = create_struct(right_data, 2, right_shape);
  Array *output = create_empty_struct(2, left_shape);

  matrix_multiplication(output, left, right);
  assert_array_equal(output, expected, 4);

  delete_struct(left);
  delete_struct(right);
  delete_struct(output);
}

void test_transpose_2d() {
  printf("test_transpose_2d\n");
  int input_shape[] = {2, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int output_shape[] = {3, 2};
  float expected[] = {1, 4, 2, 5, 3, 6};

  Array *input = create_struct(input_data, 2, input_shape);
  Array *output = create_empty_struct(2, output_shape);

  transpose(output, input, NULL);
  assert_array_equal(output, expected, 6);

  delete_struct(input);
  delete_struct(output);
}

void test_reduce_sum_axis0() {
  printf("test_reduce_sum_axis0\n");
  int input_shape[] = {3, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  c_tuple axes = {1, (int[]){0}};
  int output_shape[] = {2};
  float expected[] = {9, 12};

  Array *input = create_struct(input_data, 2, input_shape);
  Array *output = create_empty_struct(1, output_shape);

  reduce_sum(input, output, &axes);
  assert_array_equal(output, expected, 2);

  delete_struct(input);
  delete_struct(output);
}

void test_relu() {
  printf("test_relu\n");
  int shape[] = {4};
  float input_data[] = {-1, 2, -3, 4};
  float expected[] = {0, 2, 0, 4};

  Array *input = create_struct(input_data, 1, shape);
  Array *output = create_empty_struct(1, shape);

  relu(output, input);
  assert_array_equal(output, expected, 4);

  delete_struct(input);
  delete_struct(output);
}

void test_squared_error_broadcast() {
  printf("test_squared_error_broadcast\n");
  int input_shape[] = {2};
  float input_data[] = {2, 3};
  int target_shape[] = {1};
  float target_data[] = {5};
  float expected[] = {9, 4};

  Array *input = create_struct(input_data, 1, input_shape);
  Array *target = create_struct(target_data, 1, target_shape);
  Array *output = create_empty_struct(1, input_shape);

  squared_error(output, input, target);
  assert_array_equal(output, expected, 2);

  delete_struct(input);
  delete_struct(target);
  delete_struct(output);
}

void test_scalar_add() {
  printf("test_scalar_add\n");
  int shape[] = {3};
  float input_data[] = {1, 2, 3};
  float scalar = 5;
  float expected[] = {6, 7, 8};

  Array *input = create_struct(input_data, 1, shape);
  Array *output = create_empty_struct(1, shape);

  scalar_add(output, input, scalar);
  assert_array_equal(output, expected, 3);

  delete_struct(input);
  delete_struct(output);
}

void test_add_grad_broadcast() {
  printf("test_add_grad_broadcast\n");
  // Forward: add (2x3) + (3) with broadcast
  int left_shape[] = {2, 3};
  float left_data[] = {1, 2, 3, 4, 5, 6};
  int right_shape[] = {3};
  float right_data[] = {10, 20, 30};

  Array *left = create_struct(left_data, 2, left_shape);
  Array *right = create_struct(right_data, 1, right_shape);
  Array *output = create_empty_struct(2, left_shape);
  add(output, left, right); // Output is 2x3

  // Gradient of output is all ones
  Array *grad = create_full_struct(1.0f, 2, left_shape);
  Array *left_grad = create_full_struct(0.0f, 2, left_shape);
  Array *right_grad = create_full_struct(0.0f, 1, right_shape);

  add_grad(grad, 0, output, left, right, left_grad, right_grad);
  add_grad(grad, 1, output, left, right, left_grad, right_grad);

  // left_grad should be all ones (same as grad)
  float expected_left_grad[] = {1, 1, 1, 1, 1, 1};
  assert_array_equal(left_grad, expected_left_grad, 6);

  // right_grad should be summed over the broadcasted dimension (axis 0)
  float expected_right_grad[] = {2, 2, 2}; // Sum two rows
  assert_array_equal(right_grad, expected_right_grad, 3);

  delete_struct(left);
  delete_struct(right);
  delete_struct(output);
  delete_struct(grad);
  delete_struct(left_grad);
  delete_struct(right_grad);
}

void test_subtract_broadcast() {
  printf("test_subtract_broadcast\n");
  int left_shape[] = {2, 3};
  float left_data[] = {5, 5, 5, 10, 10, 10};
  int right_shape[] = {3};
  float right_data[] = {1, 2, 3};
  float expected[] = {4, 3, 2, 9, 8, 7};

  Array *left = create_struct(left_data, 2, left_shape);
  Array *right = create_struct(right_data, 1, right_shape);
  Array *output = create_empty_struct(2, left_shape);

  subtract(output, left, right);
  assert_array_equal(output, expected, 6);

  delete_struct(left);
  delete_struct(right);
  delete_struct(output);
}

void test_scalar_multiply() {
  printf("test_scalar_multiply\n");
  int shape[] = {2, 2};
  float input_data[] = {1, 2, 3, 4};
  float scalar = 2.5;
  float expected[] = {2.5, 5.0, 7.5, 10.0};

  Array *input = create_struct(input_data, 2, shape);
  Array *output = create_empty_struct(2, shape);

  scalar_multiply(output, input, scalar);
  assert_array_equal(output, expected, 4);

  delete_struct(input);
  delete_struct(output);
}

void test_scalar_subtract() {
  printf("test_scalar_subtract\n");
  int shape[] = {3};
  float input_data[] = {5, 8, 12};
  float scalar = 3;
  float expected[] = {2, 5, 9};

  Array *input = create_struct(input_data, 1, shape);
  Array *output = create_empty_struct(1, shape);

  scalar_subtract(output, input, scalar);
  assert_array_equal(output, expected, 3);

  delete_struct(input);
  delete_struct(output);
}

void test_transpose_3d() {
  printf("test_transpose_3d\n");
  int input_shape[] = {2, 3, 4};
  float input_data[24];
  for (int i = 0; i < 24; i++)
    input_data[i] = i;
  c_tuple axes = {3, (int[]){2, 0, 1}};
  int output_shape[] = {4, 2, 3};

  Array *input = create_struct(input_data, 3, input_shape);
  Array *output = create_empty_struct(3, output_shape);

  transpose(output, input, &axes);

  // Verify specific positions
  assert(fabs(output->data[output->strides[0] * 1 + output->strides[1] * 0 +
                           output->strides[2] * 0] -
              1.0) < FLOAT_TOLERANCE);
  assert(fabs(output->data[output->strides[0] * 2 + output->strides[1] * 1 +
                           output->strides[2] * 1] -
              18.0) < FLOAT_TOLERANCE);

  delete_struct(input);
  delete_struct(output);
}

void test_matrix_multiplication_batched() {
  printf("test_matrix_multiplication_batched\n");
  int left_shape[] = {3, 2, 4};
  float left_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  int right_shape[] = {4, 5};
  float right_data[] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                        0, 0, 1, 0, 0, 0, 0, 0, 1, 0};
  float expected[] = {1,  2,  3,  4,  0, 5,  6,  7,  8,  0, 9,  10, 11, 12, 0,
                      13, 14, 15, 16, 0, 17, 18, 19, 20, 0, 21, 22, 23, 24, 0};

  Array *left = create_struct(left_data, 3, left_shape);
  Array *right = create_struct(right_data, 2, right_shape);
  Array *output = create_empty_struct(3, (int[]){3, 2, 5});

  matrix_multiplication(output, left, right);
  assert_array_equal(output, expected, 30);

  delete_struct(left);
  delete_struct(right);
  delete_struct(output);
}

void test_reduce_mean_axis0() {
  printf("test_reduce_mean_axis0\n");
  int input_shape[] = {2, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  float expected[] = {2.5, 3.5, 4.5};

  Array *input = create_struct(input_data, 2, input_shape);
  Array *output = create_empty_struct(1, (int[]){2});
  c_tuple axes = {1, (int[]){0}};

  reduce_mean(output, input, &axes, NULL);
  assert_array_equal(output, expected, 2);

  delete_struct(input);
  delete_struct(output);
}

void test_reduce_mean_axis1() {
  printf("test_reduce_mean_axis1\n");
  int input_shape[] = {2, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  float axes_data[] = {1};
  float expected[] = {2.0, 5.0};

  Array *input = create_struct(input_data, 2, input_shape);
  Array *output = create_empty_struct(1, (int[]){2});
  c_tuple axes = {1, (int[]){1}};

  reduce_mean(output, input, &axes, NULL);
  assert_array_equal(output, expected, 2);

  delete_struct(input);
  delete_struct(output);
}

void test_multiplication_grad_broadcast() {
  printf("test_multiplication_grad_broadcast\n");
  // Forward: (2x3) * (3) with broadcast
  int left_shape[] = {2, 3};
  float left_data[] = {1, 2, 3, 4, 5, 6};
  int right_shape[] = {3};
  float right_data[] = {2, 3, 4};

  Array *left = create_struct(left_data, 2, left_shape);
  Array *right = create_struct(right_data, 1, right_shape);
  Array *output = create_empty_struct(2, left_shape);
  multiplication(output, left, right); // Element-wise multiplication

  // Gradient of output
  Array *grad = create_full_struct(1.0f, 2, left_shape);
  Array *left_grad = create_full_struct(0.0f, 2, left_shape);
  Array *right_grad = create_full_struct(0.0f, 1, right_shape);

  multiplication_grad(grad, 0, output, left, right, left_grad, right_grad);
  multiplication_grad(grad, 1, output, left, right, left_grad, right_grad);

  // left_grad should be grad * right (broadcasted)
  float expected_left_grad[] = {2, 3, 4, 2, 3, 4};
  assert_array_equal(left_grad, expected_left_grad, 6);

  // right_grad should be sum over broadcasted dim (axis 0)
  float expected_right_grad[] = {1 + 4, 2 + 5, 3 + 6}; // Sum columns
  assert_array_equal(right_grad, expected_right_grad, 3);

  delete_struct(left);
  delete_struct(right);
  delete_struct(output);
  delete_struct(grad);
  delete_struct(left_grad);
  delete_struct(right_grad);
}

void test_relu_grad() {
  printf("test_relu_grad\n");
  int shape[] = {4};
  float input_data[] = {-1, 2, -3, 4};
  Array *input = create_struct(input_data, 1, shape);

  // Forward pass
  Array *output = create_empty_struct(1, shape);
  relu(output, input); // [0,2,0,4]

  // Backward pass
  Array *output_grad = create_full_struct(1.0f, 1, shape);
  Array *input_grad = create_full_struct(0.0f, 1, shape);
  relu_grad(output_grad, 0, output, input, input_grad);

  float expected[] = {0, 1, 0, 1};
  assert_array_equal(input_grad, expected, 4);

  delete_struct(input);
  delete_struct(output);
  delete_struct(output_grad);
  delete_struct(input_grad);
}

void test_reduce_mean_grad() {
  printf("test_reduce_mean_grad\n");
  int input_shape[] = {2, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  Array *input = create_struct(input_data, 2, input_shape);

  // Forward pass (reduce along axis 0)
  Array *output = create_empty_struct(1, (int[]){3});
  c_tuple axes = {1, (int[]){0}};
  reduce_mean(input, output, &axes, false);

  // Backward pass
  Array *output_grad = create_struct((float[]){1, 1, 1}, 1, (int[]){3});
  Array *input_grad = create_full_struct(0.0f, 2, input_shape);
  reduce_mean_grad(output_grad, 0, output, input, &axes, false, input_grad);

  // Each element in input_grad should be 1/2 (since 2 elements averaged)
  float expected[] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  assert_array_equal(input_grad, expected, 6);

  delete_struct(input);
  delete_struct(output);
  delete_struct(output_grad);
  delete_struct(input_grad);
}

int main() {
  test_add_same_shape();
  test_add_broadcast();
  test_matrix_multiplication_2d();
  test_transpose_2d();
  test_reduce_sum_axis0();
  test_relu();
  test_scalar_add();
  test_add_grad_broadcast();
  test_squared_error_broadcast();
  test_subtract_broadcast();
  test_scalar_multiply();
  test_scalar_subtract();
  test_transpose_3d();
  test_matrix_multiplication_batched();
  test_reduce_mean_axis0();
  test_reduce_mean_axis1();
  test_multiplication_grad_broadcast();
  test_relu_grad();
  test_reduce_mean_grad();

  printf("All tests passed!\n");
  return 0;
}
