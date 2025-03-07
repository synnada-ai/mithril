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

#ifndef ARRAY_H
#define ARRAY_H

#include <string.h>
#include <stdlib.h>

typedef struct
{
    float *data;
    int *shape;
    int *strides;

    int ndim;
    int size;
} Array;

Array *create_struct(float *data, int ndim, int *shape);
Array *create_empty_struct(int ndim, int *shape);
Array *create_full_struct(float value, int ndim, int *shape);

void delete_struct(Array *s);

#endif
