#pragma once

#include "context.h"

unsigned long long gcd_u64(unsigned long long a, unsigned long long b);

unsigned long long phi_u64(unsigned long long n);

unsigned long long real_shape_words(unsigned long long P, int S, bool *ok);

bool real_run_geometry(unsigned long long p,int bits,ecm_stage2::Geometry &g);

bool real_run_words(unsigned long long P,int S,unsigned long long *out_words,
                           unsigned long long *n_fold,unsigned long long *n_tree);
