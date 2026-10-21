#pragma once
#include <gmp.h>
#include <cstdint>
#include <cstddef>

// Checked Atkin-Morain construction, a=1. All inverses and singularities are
// classified modulo the actual target, including when Stage2 uses a carrier.
namespace ecm_edwards {
enum class Setup { Ready, Factor, Degenerate };
struct Integer {
    mpz_t v;
    Integer() { mpz_init(v); }
    ~Integer() { mpz_clear(v); }
    Integer(const Integer &) = delete;
    Integer &operator=(const Integer &) = delete;
};
inline Setup inverse(mpz_t out, const mpz_t value, const mpz_t n, mpz_t factor) {
    if (mpz_invert(out,value,n)) return Setup::Ready;
    mpz_gcd(factor,value,n);
    return mpz_cmp(factor,n)<0 ? Setup::Factor : Setup::Degenerate;
}

inline Setup construct(mpz_t d, mpz_t px, mpz_t py, mpz_t a24,
                       mpz_t factor, uint64_t sigma, const mpz_t n) {
    mpz_set_ui(factor,0);
    if (!sigma || mpz_cmp_ui(n,3)<=0 || !mpz_odd_p(n)) return Setup::Degenerate;
    Integer X,Y,Z,A,B,C,S,M,U,V,H,J,K,T,inv,s,t,alpha,beta,h,num,den;
    auto mul=[&](mpz_t r,const mpz_t a,const mpz_t b) {mpz_mul(r,a,b);mpz_mod(r,r,n);};
    auto sqr=[&](mpz_t r,const mpz_t a) {mul(r,a,a);};
    auto reduce=[&](mpz_t r) {mpz_mod(r,r,n);};
    // Jacobian scalar multiplication on t^2=s^3-8s-32 avoids unchecked
    // affine inverses in the auxiliary Weierstrass curve.
    mpz_set_ui(Z.v,0);
    auto double_point=[&]() {
        if (mpz_sgn(Z.v)) {
            sqr(A.v,X.v); sqr(B.v,Y.v); sqr(C.v,B.v);
            mul(S.v,X.v,B.v); mpz_mul_ui(S.v,S.v,4); reduce(S.v);
            sqr(M.v,Z.v); sqr(M.v,M.v); mpz_mul_ui(M.v,M.v,8);
            mpz_mul_ui(T.v,A.v,3); mpz_sub(M.v,T.v,M.v); reduce(M.v);
            sqr(U.v,M.v); mpz_sub(U.v,U.v,S.v); mpz_sub(U.v,U.v,S.v); reduce(U.v);
            mpz_sub(V.v,S.v,U.v); mul(V.v,M.v,V.v);
            mpz_mul_ui(T.v,C.v,8); mpz_sub(V.v,V.v,T.v); reduce(V.v);
            mul(Z.v,Y.v,Z.v); mpz_mul_ui(Z.v,Z.v,2); reduce(Z.v);
            mpz_set(X.v,U.v); mpz_set(Y.v,V.v);
        }
    };
    for (int bit=63;bit>=0;--bit) {
        double_point();
        if (!((sigma>>bit)&1)) continue;
        if (!mpz_sgn(Z.v)) {mpz_set_ui(X.v,12);reduce(X.v);mpz_set_ui(Y.v,40);reduce(Y.v);mpz_set_ui(Z.v,1);continue;}
        sqr(A.v,Z.v); mpz_mul_ui(U.v,A.v,12); reduce(U.v);
        mul(B.v,A.v,Z.v); mpz_mul_ui(V.v,B.v,40); reduce(V.v);
        mpz_sub(H.v,U.v,X.v); reduce(H.v);
        mpz_sub(J.v,V.v,Y.v); reduce(J.v);
        if (!mpz_sgn(H.v)) {
            if(!mpz_sgn(J.v)){double_point();continue;}
            mpz_gcd(factor,J.v,n);
            if(mpz_cmp_ui(factor,1)>0 && mpz_cmp(factor,n)<0)return Setup::Factor;
            mpz_set_ui(Z.v,0);continue;
        }
        sqr(A.v,H.v); mul(B.v,A.v,H.v); mul(K.v,X.v,A.v);
        sqr(U.v,J.v); mpz_sub(U.v,U.v,B.v);mpz_sub(U.v,U.v,K.v);mpz_sub(U.v,U.v,K.v);reduce(U.v);
        mpz_sub(V.v,K.v,U.v);mul(V.v,J.v,V.v);mul(T.v,Y.v,B.v);mpz_sub(V.v,V.v,T.v);reduce(V.v);
        mul(Z.v,Z.v,H.v);mpz_set(X.v,U.v);mpz_set(Y.v,V.v);
    }
    Setup rc=inverse(inv.v,Z.v,n,factor); if(rc!=Setup::Ready)return rc;
    sqr(T.v,inv.v);mul(s.v,X.v,T.v);mul(T.v,T.v,inv.v);mul(t.v,Y.v,T.v);
    // alpha=(s-9)/(t+s+16).
    mpz_sub_ui(num.v,s.v,9);mpz_add(den.v,t.v,s.v);mpz_add_ui(den.v,den.v,16);
    rc=inverse(inv.v,den.v,n,factor);if(rc!=Setup::Ready)return rc;mul(alpha.v,num.v,inv.v);
    // beta=2*alpha*(4*alpha+1)/(8*alpha^2-1).
    mpz_mul_ui(num.v,alpha.v,4);mpz_add_ui(num.v,num.v,1);mul(num.v,num.v,alpha.v);mpz_mul_ui(num.v,num.v,2);
    sqr(den.v,alpha.v);mpz_mul_ui(den.v,den.v,8);mpz_sub_ui(den.v,den.v,1);
    rc=inverse(inv.v,den.v,n,factor);if(rc!=Setup::Ready)return rc;mul(beta.v,num.v,inv.v);
    mpz_mul_ui(h.v,beta.v,2);mpz_sub_ui(h.v,h.v,1);reduce(h.v);
    sqr(T.v,h.v);mpz_mul_ui(num.v,T.v,2);mpz_sub_ui(num.v,num.v,1);sqr(den.v,T.v);
    rc=inverse(inv.v,den.v,n,factor);if(rc!=Setup::Ready)return rc;mul(d,num.v,inv.v);
    // The xDBL convention is (A+2)/4=1/(1-d); Prime95 stores its reciprocal.
    mpz_ui_sub(den.v,1,d);mul(T.v,d,den.v);mpz_gcd(factor,T.v,n);
    if(mpz_cmp_ui(factor,1)!=0)return mpz_cmp(factor,n)<0?Setup::Factor:Setup::Degenerate;
    rc=inverse(a24,den.v,n,factor);if(rc!=Setup::Ready)return rc;
    mpz_mul_ui(num.v,beta.v,4);mpz_sub_ui(num.v,num.v,3);mul(num.v,num.v,h.v);
    mpz_mul_ui(den.v,beta.v,6);mpz_sub_ui(den.v,den.v,5);
    rc=inverse(inv.v,den.v,n,factor);if(rc!=Setup::Ready)return rc;mul(px,num.v,inv.v);
    sqr(num.v,t.v);mpz_mul_ui(T.v,t.v,50);mpz_add(num.v,num.v,T.v);
    sqr(T.v,s.v);mul(U.v,T.v,s.v);mpz_mul_ui(U.v,U.v,2);mpz_sub(num.v,num.v,U.v);
    mpz_mul_ui(T.v,T.v,27);mpz_add(num.v,num.v,T.v);mpz_sub_ui(num.v,num.v,104);mul(num.v,num.v,h.v);
    mpz_mul_ui(den.v,s.v,3);mpz_add(den.v,den.v,t.v);mpz_sub_ui(den.v,den.v,2);
    mpz_add(T.v,t.v,s.v);mpz_add_ui(T.v,T.v,16);mul(den.v,den.v,T.v);
    rc=inverse(inv.v,den.v,n,factor);if(rc!=Setup::Ready)return rc;mul(py,num.v,inv.v);
    sqr(U.v,px);sqr(V.v,py);mpz_add(T.v,U.v,V.v);mpz_sub_ui(T.v,T.v,1);
    mul(U.v,U.v,V.v);mul(U.v,U.v,d);mpz_sub(T.v,T.v,U.v);reduce(T.v);
    if(mpz_sgn(T.v))return Setup::Degenerate;
    mpz_set_ui(factor,0);return Setup::Ready;
}
inline bool point_valid(const mpz_t n,const mpz_t d,mpz_t *p) {
    for(unsigned j=0;j<4;++j)if(mpz_sgn(p[j])<0 || mpz_cmp(p[j],n)>=0)return false;
    if(!mpz_sgn(p[0]) && !mpz_sgn(p[1]) && !mpz_sgn(p[2]) && !mpz_sgn(p[3]))return false;
    Integer a,b,c;
    mpz_mul(a.v,p[0],p[1]);mpz_mul(b.v,p[2],p[3]);mpz_sub(a.v,a.v,b.v);mpz_mod(a.v,a.v,n);
    if(mpz_sgn(a.v))return false;
    mpz_mul(a.v,p[0],p[0]);mpz_mul(b.v,p[1],p[1]);mpz_add(a.v,a.v,b.v);
    mpz_mul(b.v,p[2],p[2]);mpz_sub(a.v,a.v,b.v);
    mpz_mul(c.v,p[3],p[3]);mpz_mul(c.v,c.v,d);mpz_sub(a.v,a.v,c.v);mpz_mod(a.v,a.v,n);
    return !mpz_sgn(a.v);
}
class Array {
    mpz_t *p_;size_t size_;
public:
    explicit Array(size_t size):p_(new mpz_t[size]),size_(size){for(size_t i=0;i<size_;++i)mpz_init(p_[i]);}
    ~Array(){for(size_t i=0;i<size_;++i)mpz_clear(p_[i]);delete[] p_;}
    Array(const Array &)=delete;Array &operator=(const Array &)=delete;
    mpz_t *data(){return p_;}
    mpz_t &operator[](size_t i){return p_[i];}
};
} // namespace ecm_edwards
