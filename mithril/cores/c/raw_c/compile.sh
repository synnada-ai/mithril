#!/bin/bash
cc ops.c array.c utils.c -shared -fPIC -g -o libmithrilc.so
