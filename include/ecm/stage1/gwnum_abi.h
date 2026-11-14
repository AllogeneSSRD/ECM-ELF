#ifndef ECM_STAGE1_GWNUM_ABI_H
#define ECM_STAGE1_GWNUM_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECM_GWNUM_ABI_VERSION 1u
#define ECM_GWNUM_PLUGIN_FILENAME L"gwnum.dll"
#define ECM_GWNUM_GET_API_SYMBOL "ecm_gwnum_get_api"

/* All integers use exactly words little-endian 32-bit limbs in caller-owned
 * buffers. Modulus and coordinates must be unsigned, with residues below the
 * modulus. A zero exponent requests a general modulus plan. A nonzero exponent
 * is the exact p in 2^p-1. To start a new ladder, omit both X1 and Z1; to resume,
 * supply both. The other pointers must be present. */
typedef struct ecm_gwnum_api {
    uint32_t abi_version;
    uint32_t struct_size;
    const char* gwnum_version;
    void* (*create)(const uint32_t* modulus, uint32_t words, uint32_t mersenne_exponent,
                    const uint32_t* a24, const uint32_t* xdiff, const uint32_t* x0,
                    const uint32_t* z0, const uint32_t* x1, const uint32_t* z1,
                    char* error, uint32_t error_size);
    /* One byte per scalar bit, in execution order. Zero denotes success. */
    int (*advance)(void* context, const uint8_t* bits, size_t count,
                   char* error, uint32_t error_size);
    /* Read the current pair or omit both X1 and Z1 to read only the first point.
     * A nonzero return leaves all output buffers unspecified. */
    int (*read)(void* context, uint32_t* x0, uint32_t* z0, uint32_t* x1, uint32_t* z1,
                char* error, uint32_t error_size);
    void (*destroy)(void* context);
} ecm_gwnum_api;

/* The getter returns NULL for unsupported ABI versions. The table and version
 * string belong to the DLL and remain valid while the DLL is loaded. */
typedef const ecm_gwnum_api* (*ecm_gwnum_get_api_fn)(uint32_t abi_version);

#ifdef __cplusplus
}
#endif

#endif
