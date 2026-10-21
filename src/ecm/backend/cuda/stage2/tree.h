#pragma once

#include "product.h"

struct FTreeStats {
    size_t leaves = 0, padded = 0;
    unsigned long long muls = 0;
    unsigned long long node_peak_bytes = 0, node_retained_bytes = 0, node_released_bytes = 0;
    unsigned long long nodes_released = 0, passthrough_moves = 0;
    double t_release = 0.0;
};

struct GMemoryStats {
    unsigned long long pinned_trees=0,pinned_slices=0,pinned_words=0,pageable_words=0;
    unsigned long long leaf_fallbacks=0,legacy_trees=0,pinned_borrow_peak_bytes=0;
    unsigned long long rawA_peak_bytes=0,rawB_peak_bytes=0;
};

struct GRootStats {
    unsigned long long builds = 0, nodes_released = 0, passthrough_moves = 0;
    unsigned long long peak_node_bytes = 0, peak_retained_bytes = 0, released_bytes = 0;
    unsigned long long input_released_bytes = 0, root_words = 0;
    unsigned long long root_hash = 1469598103934665603ull;
    bool root_hash_complete=true;
    double t_release = 0.0;
};

std::vector<std::vector<unsigned long long>> build_tree_flat(
    PolyLayer &L, const std::vector<std::vector<unsigned long long>> &leaf,
    std::vector<size_t> &deg, size_t &pad_out, FTreeStats &fs, int cat = -1,
    bool keep_children = true);

std::vector<unsigned long long> groot_product_gmp(PolyLayer &L,
    const std::vector<unsigned long long> &a,const std::vector<unsigned long long> &b);

std::vector<std::vector<unsigned long long>> build_groot_device(
    PolyLayer &L,const std::vector<std::vector<unsigned long long>> &leaf,
    std::vector<size_t> &deg,size_t &pad_out,FTreeStats &fs,int cat,
    size_t device_n=0,const std::function<void(unsigned long long*)> &fill={},
    const std::function<void(const unsigned long long*,size_t)> &root_sink={},bool balanced=false);

std::vector<std::vector<unsigned long long>> build_groot_select(
    PolyLayer &L,const std::vector<std::vector<unsigned long long>> &leaf,
    std::vector<size_t> &deg,size_t &pad,FTreeStats &fs,int cat,bool keep_children,
    const std::function<void(const unsigned long long*,size_t)> &root_sink={},bool balanced=false);

void groot_input_digest_host(const std::vector<unsigned long long> &input,
    unsigned long long &words,unsigned long long *digest);

extern GRootStats g_groot;

extern GMemoryStats g_gmemory;
