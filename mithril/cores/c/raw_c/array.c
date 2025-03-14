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

#include "array.h"

/* Create a new struct with the given data, ndim and shape */
Array *create_struct(float *data, int ndim, int *shape)
{
    Array *s = (Array *)malloc(sizeof(Array));
    s->ndim = ndim;
    s->shape = (int *)malloc(ndim * sizeof(int));
    s->strides = (int *)malloc(ndim * sizeof(int));

    int total_size = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        s->strides[i] = total_size;
        s->shape[i] = shape[i];
        total_size *= shape[i];
    }

    s->size = total_size;
    s->data = (float *)malloc(total_size * sizeof(float));
    memcpy(s->data, data, total_size * sizeof(float));

    return s;
}

/* Create a new struct with the given ndim and shape */
Array *create_empty_struct(int ndim, int *shape)
{
    Array *s = (Array *)malloc(sizeof(Array));
    s->ndim = ndim;
    s->shape = (int *)malloc(ndim * sizeof(int));
    s->strides = (int *)malloc(ndim * sizeof(int));

    int total_size = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        s->strides[i] = total_size;
        s->shape[i] = shape[i];
        total_size *= shape[i];
    }

    s->size = total_size;
    s->data = (float *)malloc(total_size * sizeof(float));

    return s;
}

/* Create a new struct filled with the given value, ndim and shape */
Array *create_full_struct(float value, int ndim, int *shape)
{
    Array *s = (Array *)malloc(sizeof(Array));
    s->ndim = ndim;
    s->shape = (int *)malloc(ndim * sizeof(int));
    s->strides = (int *)malloc(ndim * sizeof(int));

    int total_size = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        s->strides[i] = total_size;
        s->shape[i] = shape[i];
        total_size *= shape[i];
    }

    s->size = total_size;
    s->data = (float *)malloc(total_size * sizeof(float));

    for (size_t i = 0; i < total_size; i++)
    {
        s->data[i] = value;
    }

    return s;
}


void delete_struct(Array *s)
{
    free(s->data);
    free(s->shape);
    free(s->strides);
    free(s);
}