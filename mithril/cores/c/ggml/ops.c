#include "ops.h"

struct ggml_tensor * add(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * res = ggml_add(ctx, left, right);
    return res;
}

struct ggml_tensor * multiplication(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * res = ggml_mul(ctx, left, right);
    return res;
}