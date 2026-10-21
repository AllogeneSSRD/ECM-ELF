#pragma once

#include "context.h"

struct PolyLayer;

struct SmallPrimeBabyCache;

struct GiantSeedStats {
    unsigned long long chunks=0,points=0,avoided_d2h_bytes=0,avoided_h2d_bytes=0;
    unsigned long long avoided_cpu_modmuls=0,avoided_montmuls=0,checked_words=0;
    unsigned long long segments=0,segment_checks=0,segment_fix_muls=0,fix_table_peak_bytes=0;
    unsigned long long base_builds=0,base_nonunits=0,paired_chunks=0,paired_ladders=0;
    unsigned long long base_cpu_builds=0,base_gpu_builds=0,base_h2d_bytes=0,base_checked_words=0;
    double base_cpu_seconds=0,base_build_seconds=0;
    unsigned long long scalar_h2d_avoided=0,base_d2h_bytes=0;
};

struct GfinvStats {
    unsigned long long requests=0,cache_hits=0,groups=0,segments=0,group_attempts=0,
        group_failures=0,individual_attempts=0,good=0,nonunits=0,scratch_peak_bytes=0;
    double t_prepare=0;
};

extern GfinvStats g_gfinv;
struct GfinvBatch {
    static constexpr size_t GROUP=64;
    const std::vector<unsigned long long> &products;
    size_t W,base=(size_t)-1,count=0;
    mpz_srcptr N;
    bool enabled,unit[GROUP]{};
    mpz_t prefix[GROUP+1],inverse[GROUP],acc,value;
    GfinvStats &stats;
    GfinvBatch(const std::vector<unsigned long long> &p,size_t w,mpz_srcptr n,bool on,
               GfinvStats &s=g_gfinv);
    GfinvBatch(const GfinvBatch &)=delete;
    GfinvBatch &operator=(const GfinvBatch &)=delete;
    ~GfinvBatch();
    void prepare(size_t index);
    bool get(mpz_t out,size_t index);
};

struct S3Workspace {
    const LadderCtx *C = nullptr;
    unsigned long long *dn = nullptr, *dqx = nullptr, *dqz = nullptr, *da24 = nullptr,
                       *dmone = nullptr;
    unsigned long long *djs = nullptr;
    unsigned long long *dx = nullptr, *dz = nullptr;
    unsigned long long *giant_base=nullptr;
    unsigned long long giant_base_d=0;
    bool giant_base_unit=false;
    unsigned long long *dvals = nullptr, *dprod = nullptr;
    unsigned long long *dsegfix=nullptr;
    size_t segfix_cap=0;
    size_t js_cap = 0, pt_cap = 0, val_cap = 0, prod_cap = 0;
    unsigned long long ladder_calls = 0, ladder_points_total = 0, prod_launches = 0;
    size_t bytes = 0;

    ~S3Workspace();
    void init(const LadderCtx &c);
    void need_segfix(size_t seg);
    bool need_giant_base(unsigned long long d,mpz_srcptr modulus);
    void need_pts(size_t n);
    void need_vals(size_t n);
    void need_products(size_t n);
};

struct DeviceGLeafStats {
    unsigned long long requested_chunks=0,chunks=0,fallback_chunks=0,groups=0,bad_groups=0,
        good_segments=0,device_trees=0,device_leaf_words=0,patch_words=0,
        group_d2h_bytes=0,bad_segment_d2h_bytes=0,bad_point_d2h_bytes=0,
        avoided_point_d2h_bytes=0,avoided_segment_d2h_bytes=0,avoided_leaf_h2d_bytes=0,
        coord_peak_bytes=0,group_peak_bytes=0,checked_groups=0,checked_leaf_words=0,retry_chunks=0;
    double t_prepare=0,t_invert=0,t_fill=0;
};

struct ResidentGiant {
    static constexpr size_t GROUP=64,SEG=S2G_GFINV_SEG;
    unsigned long long *x=nullptr,*z=nullptr,*segments=nullptr,*groups_device=nullptr,*group_fix=nullptr;
    size_t n=0,w=0;bool own=false;
    std::vector<unsigned char> group_good;
    std::vector<std::vector<unsigned long long>> bad_x,bad_z;
    ~ResidentGiant();
    const unsigned long long *coord(size_t q,bool use_z) const;
    bool good(size_t segment) const;
    void prepare(S3Workspace &ws,PolyLayer &L,std::vector<unsigned long long> &gseg,
        mpz_t Ginv,unsigned long long &gamma_points,unsigned long long &projective_segments,DeviceGLeafStats &st);
};

struct DeviceBabyStats {
    size_t points=0,groups=0,bad_groups=0,payload_bytes=0,coordinate_d2h_bytes=0,
        root_d2h_bytes=0,leaf_d2h_bytes=0,seed_h2d_bytes=0,checked_words=0;
    double ladder_seconds=0,invert_seconds=0,total_seconds=0;
};

struct BabyBuffer {
    void *ptr=nullptr;
    ~BabyBuffer();
    bool allocate(size_t bytes);
    unsigned long long *words();
};

struct SmallPrimeBabyCache {
    LadderCtx key;
    unsigned long long D=0,B1=0,B2=0;
    std::vector<unsigned long long> js;
    std::vector<std::pair<size_t,std::string>> nonunit;
    bool ready=false;
    void begin(const LadderCtx &C,unsigned long long d,unsigned long long b1,unsigned long long b2,
               const std::vector<unsigned long long> &indices);
    void record(size_t index,const mpz_t gcd);
    bool matches(const LadderCtx &C,const Stage2Params &SP) const;
    void gcd_at(size_t index,mpz_t g) const;
    size_t payload_bytes() const;
};

bool affine_x_gmp_checked(mpz_t out, const mpz_t X, const mpz_t Z, const mpz_t N);

void affine_x_gmp(mpz_t out, const mpz_t X, const mpz_t Z, const mpz_t N);

void words_neg_mod_n(std::vector<unsigned long long> &dst, size_t off,
                            const unsigned long long *x, const unsigned long long *n, size_t W);

void gfinv_segprod_host(std::vector<unsigned long long> &out, size_t npts, size_t nw,
                               const std::vector<unsigned long long> &gz, const mpz_t N);

void ladder_points(const LadderCtx &C, const std::vector<unsigned long long> &js,
                          std::vector<unsigned long long> &outx,
                          std::vector<unsigned long long> &outz);

void ladder_points_ws(S3Workspace &W, const std::vector<unsigned long long> &js,
                             std::vector<unsigned long long> &outx,
                             std::vector<unsigned long long> &outz);

void giant_ladder_into(S3Workspace &ws,const LadderCtx &C,unsigned long long D,
    unsigned long long first,size_t points,unsigned long long *x,unsigned long long *z);

bool giant_chunk_chain(PolyLayer &L, const LadderCtx &C, S3Workspace &W, unsigned long long D,
                              unsigned long long clo, unsigned long long chi,
                              unsigned long long per_block, std::vector<unsigned long long> &gx,
                              std::vector<unsigned long long> &gz, bool check,
                              unsigned long long &seed_points,
                              std::vector<unsigned long long> *gseg = nullptr,
                              unsigned long long seg = 16,ResidentGiant *resident=nullptr,bool wide_tail=false);

int suyama_curve(mpz_t a24, mpz_t px, mpz_t pz, unsigned long long sigma, const mpz_t N);

bool device_baby_generate(const LadderCtx &C,const mpz_t N,
    const std::vector<unsigned long long> &indices,
    std::vector<std::vector<unsigned long long>> &leaf,SmallPrimeBabyCache *cache,
    std::vector<std::string> &factors,unsigned long long &noninv,DeviceBabyStats &st,bool force_check=false);

extern GiantSeedStats g_giant_seed;
