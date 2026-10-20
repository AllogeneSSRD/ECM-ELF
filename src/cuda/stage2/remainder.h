#pragma once

#include "product.h"

#include "../common/plain_scale.h"

struct FoldFlatStats {
    bool enabled=false;
    unsigned long long folds=0, muls=0, sub_coeffs=0, peak_bytes=0;
    double t_prepare=0, t_multiply=0, t_subtract=0, t_bridge=0;
};

struct ScaledRootDeviceStats {
    bool requested=false,enabled=false;
    size_t coefficients=0;
    unsigned long long h2d_bytes=0,d2h_bytes=0,check_d2h_bytes=0,checked_words=0;
    unsigned long long avoided_h2d_bytes=0,avoided_h_readback_bytes=0;
    double seconds=0;
    const char *fallback="none";
};

struct FoldDeviceStats {
    bool requested=false,enabled=false;
    unsigned reuse=kFoldOwnerReuse;
    unsigned long long layout_bytes=0,saved_bytes=0;
    std::string fallback="none";
    unsigned long long folds=0,muls=0,sub_coeffs=0,peak_bytes=0,h2d_bytes=0,d2h_bytes=0;
    unsigned long long avoided_h2d_bytes=0,avoided_d2h_bytes=0,checked_words=0;
    bool root_requested=false;
    unsigned long long root_device_trees=0,root_device_words=0,root_checked_words=0;
    unsigned long long root_h2d_avoided=0,root_d2h_avoided=0,root_digest_words=0,root_digest[2]={0,0};
    double root_t_handoff=0;
    double t_setup=0,t_upload=0,t_reverse=0,t_subtract=0,t_readback=0;
};

struct FoldDeviceState {
    S4ResidentOwner memory;
    unsigned long long *map=nullptr,*length=nullptr,*modulus=nullptr,*digest=nullptr;
    PolyLayer *layer=nullptr;FoldDeviceStats *stats=nullptr;FoldFlatStats *flat=nullptr;
    size_t P=0,W=0,hcount=0,gcount=0,f=0,inv=0,h=0,g=0,reverse=0,t=0,q=0,qb=0;
    bool active=false,check=false;
    const std::vector<unsigned long long> *host_F=nullptr;
    ~FoldDeviceState();
    void release();
    bool init(PolyLayer &L,const std::vector<unsigned long long> &F,
              const std::vector<unsigned long long> &inverse,FoldDeviceStats &st,FoldFlatStats &fs);
    void upload(size_t offset,const std::vector<unsigned long long> &v);
    void digest_root(size_t offset,size_t words);
    void read_digest();
    void accept_device_root(const unsigned long long *src,size_t count,bool seed_root);
    void seed(const std::vector<unsigned long long> &H);
    std::vector<unsigned long long> read(size_t offset,size_t count);
    void mul(size_t a,size_t ma,size_t b,size_t mb,size_t dest,size_t count);
    void reverse_copy(size_t src,size_t top,size_t count);
    void step(const std::vector<unsigned long long> &G);
    void step_loaded(size_t ng,const std::vector<unsigned long long> *host_G=nullptr);
    void scale(mpz_srcptr factor,PlainScaleStats &st,bool verify,bool poison);
    void scaled_root(std::vector<unsigned long long> &root,ScaledRootDeviceStats &st,
                     bool verify,bool poison,bool keep_resident=false);
    void finish(std::vector<unsigned long long> &H);
};

struct ScaledStats {
    unsigned long long levels=0, mul_calls=0, mul_pairs=0, copies=0, zeros=0;
    unsigned long long states=0, words=0, leaves=0, checked_states=0, checked_words=0;
    unsigned long long frontier_peak_bytes=0, pack_peak_bytes=0, root_inverse_reused=0;
    unsigned long long root_divisions=0;
};

struct ScaledFrontierDevice {
    FoldDeviceState *fold=nullptr;
    unsigned long long *metadata=nullptr;
    bool requested=false,enabled=false;
    const char *fallback="none";
    size_t metadata_words=0;
    unsigned long long metadata_bytes=0,owner_and_metadata_bytes=0;
    unsigned long long f_h2d_bytes=0,metadata_h2d_bytes=0,leaf_d2h_bytes=0,check_d2h_bytes=0;
    unsigned long long avoided_parent_h2d_bytes=0,avoided_state_d2h_bytes=0;
    unsigned long long logical_frontier_peak_bytes=0,host_staging_peak_bytes=0,pinned_borrow_peak_bytes=0;
    double setup_seconds=0,upload_seconds=0;
    ~ScaledFrontierDevice();
    void release();
    bool init(FoldDeviceState &owner);
    // Input F stays in ordinary order on the host. The reverse gather packs it
    // directly into NTT digits. Existing pinned output slots are borrowed only
    // after their recorded D2H/H2D consumer has completed.
    void upload_siblings(unsigned long long *dst,const std::vector<std::vector<unsigned long long>> &Ft,
        size_t nbase,const std::vector<std::pair<size_t,size_t>> &group,size_t mb,size_t w);
    void run(PolyLayer &L,const std::vector<std::vector<unsigned long long>> &Ft,
        const std::vector<size_t> &deg,size_t pad,const CPoly &H,
        std::vector<std::vector<unsigned long long>> &values,ScaledStats &st,int cat,bool check);
    void print()const;
};

std::vector<unsigned long long> cp_zero(size_t W);

void cp_resize(CPoly &a, size_t n, size_t W);

CPoly cp_from_flat(const std::vector<unsigned long long> &f, size_t deg, size_t W);

std::vector<unsigned long long> cp_to_flat(const CPoly &p, size_t W);

bool cp_coeff_zero(const std::vector<unsigned long long> &c);

void cp_trim(CPoly &a);

bool cp_is_one(const CPoly &a, size_t W);

void cp_coeff_sub(std::vector<unsigned long long> &dst,
                         const std::vector<unsigned long long> &x,
                         const std::vector<unsigned long long> &y, const mpz_t N, size_t W);

CPoly cp_addsub(const CPoly &a, const CPoly &b, PolyLayer &L, bool sub);

CPoly cp_mul(const CPoly &a, const CPoly &b, PolyLayer &L, size_t keep = (size_t)-1);

CPoly cp_inv_series(const CPoly &a, size_t k, PolyLayer &L);

void cp_divmod(CPoly &q, CPoly &r, const CPoly &a, const CPoly &b, PolyLayer &L);

CPoly cp_mod(const CPoly &a, const CPoly &b, PolyLayer &L);

void leaf_mod_linear_batch(const std::vector<unsigned long long> &A, size_t da,
                                  const std::vector<unsigned long long> &B, size_t nbatch,
                                  size_t W, const mpz_t N, std::vector<unsigned long long> &out);

void fold_flat_step(PolyLayer &L, const std::vector<unsigned long long> &G,
                           std::vector<unsigned long long> &H,
                           const std::vector<unsigned long long> &F,
                           const std::vector<unsigned long long> &finv, FoldFlatStats &st);

void descent_scaled(PolyLayer &L,
    const std::vector<std::vector<unsigned long long>> &Ft,
    const std::vector<size_t> &Fdeg, size_t Fpad, const CPoly &H,
    const CPoly *cached_finv, std::vector<std::vector<unsigned long long>> &values,
    ScaledStats &st, int cat, bool check, std::vector<unsigned long long> *prepared_root=nullptr);

std::vector<unsigned long long> cp_addsub_flat(const std::vector<unsigned long long> &A,
                                                      const std::vector<unsigned long long> &B,
                                                      size_t n, size_t W, const mpz_t N,
                                                      bool sub);

void cp_coeff_sub_p(unsigned long long *dst, const unsigned long long *x,
                           const unsigned long long *y, const mpz_t N, size_t W);

void inv_series_batch(PolyLayer &L, const std::vector<unsigned long long> &A, size_t m,
                             size_t k, size_t nbatch, std::vector<unsigned long long> &g, int cat);

void divmod_batch(PolyLayer &L, const std::vector<unsigned long long> &A,
                         const std::vector<unsigned long long> &B, size_t da, size_t db,
                         size_t nbatch, std::vector<unsigned long long> &out, int cat);
