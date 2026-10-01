#ifndef HUE_TENSOR_H
#define HUE_TENSOR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Opaque owning handle. Free exactly once. Buffers must have the stated size.
   Handles and buffers may not alias output buffers or be freed concurrently.
   Status 0 succeeds, 1 rejects input, 2 rejects a storage operation, 3 catches
   a Rust panic. Output handles are NULL on failure. Paths use UTF-8.
   ht_last_error is thread-local and valid until the next API operation.
   D is sub-hue refinement, never an IEEE exponent or physical depth. */
typedef struct ht_tensor ht_tensor;
typedef struct ht_token { uint8_t r, g, b, d; } ht_token;
uint32_t ht_abi_version(void);
uint32_t ht_modulus(void);
const char *ht_last_error(void);
int32_t ht_new_phases(const uint32_t *, size_t, const size_t *, size_t, ht_tensor **);
int32_t ht_new_rgbd(const uint8_t *, size_t, const size_t *, size_t, ht_tensor **);
int32_t ht_new_rgb(const uint8_t *, size_t, const size_t *, size_t, ht_tensor **);
void ht_free(ht_tensor *);
size_t ht_len(const ht_tensor *);
size_t ht_ndim(const ht_tensor *);
int32_t ht_copy_shape(const ht_tensor *, size_t *, size_t);
int32_t ht_copy_tokens(const ht_tensor *, uint8_t *, size_t);
int32_t ht_copy_rgb(const ht_tensor *, uint8_t *, size_t);
int32_t ht_copy_phases(const ht_tensor *, uint32_t *, size_t);
/* opcode 0 adds phases, 1 subtracts, 2 multiplies, all modulo 391680. */
int32_t ht_binary(const ht_tensor *, const ht_tensor *, uint32_t, ht_tensor **);
int32_t ht_rotate(const ht_tensor *, int64_t, ht_tensor **);
int32_t ht_blend(const ht_tensor *, const ht_tensor *, uint8_t, ht_tensor **);
int32_t ht_matmul(const ht_tensor *, const ht_tensor *, ht_tensor **);
int32_t ht_dot(const ht_tensor *, const ht_tensor *, ht_tensor **);
int32_t ht_sum(const ht_tensor *, size_t, ht_tensor **);
int32_t ht_reshape(const ht_tensor *, const size_t *, size_t, ht_tensor **);
int32_t ht_permute(const ht_tensor *, const size_t *, size_t, ht_tensor **);
int32_t ht_slice(const ht_tensor *, size_t, size_t, size_t, size_t, ht_tensor **);
int32_t ht_save(const ht_tensor *, const char *);
int32_t ht_load(const char *, size_t, ht_tensor **);

#ifdef __cplusplus
}
#endif
#endif
