#!/bin/bash
cc ops.c -L. -lggml-base -shared -fPIC -o libmithrilggml.so 