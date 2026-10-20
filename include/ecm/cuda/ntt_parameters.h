#pragma once
#ifndef NTT_PROBE_NAME
#define NTT_PROBE_NAME "ntt_poly_probe"
#endif
#define GL_P 0xFFFFFFFF00000001ull
#define SLOT_MOD 4294967291ull
#define FUSE_TILE_THREADS 512
#define FUSE_OUTER_THREADS 256
#define FUSE_MAX_M 4
#define FUSE_MAX_CARRY_ROUNDS 10
#define FUSE_MAX_PASSES 8
#ifndef NTT_GL_SHORT_REDUCE_DEFAULT
#define NTT_GL_SHORT_REDUCE_DEFAULT 0
#endif
#ifndef NTT_GL_FIXED_MODE
#define NTT_GL_FIXED_MODE -1
#endif
#if NTT_GL_FIXED_MODE != -1 && NTT_GL_FIXED_MODE != 0 && NTT_GL_FIXED_MODE != 1 && NTT_GL_FIXED_MODE != 3
#error "NTT_GL_FIXED_MODE must be -1 (runtime), 0 (fold), 1 (short), or 3 (PTX)"
#endif
