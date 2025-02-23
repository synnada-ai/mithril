#ifndef MITHRIL_GGML_OPS_H
#define MITHRIL_GGML_OPS_H

#include "ggml/include/ggml.h"


struct ggml_tensor * add(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right);
struct ggml_tensor * multiplication(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right);

#endif // MITHRIL_GGML_OPS_H
