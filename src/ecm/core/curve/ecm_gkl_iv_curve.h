#pragma once
#include "ecm_edwards_curve.h"
#include <initializer_list>

// Experimental GKL 2017, Table 1 row 4 / Table 2 last row.
// a=-1, d=-e^4. Seed m is [m](5,8), NOT a Prime95 Edwards sigma alias.
// This constructor is not connected to production dispatch or save identities.
namespace ecm_gkl_iv {
using ecm_edwards::Integer;
using ecm_edwards::Setup;
inline constexpr char experiment_version[] = "1.0.0";

inline Setup construct(mpz_t d, mpz_t px, mpz_t py, mpz_t a24, mpz_t factor,
                       uint64_t m, const mpz_t n) {
    mpz_set_ui(factor,0);
    if(!m || mpz_cmp_ui(n,3)<=0 || !mpz_odd_p(n))return Setup::Degenerate;
    Integer X,Y,Z,A,B,C,S,M,U,V,H,J,K,T,inv,t,num,den,e;
    auto reduce=[&](mpz_t x){mpz_mod(x,x,n);};
    auto mul=[&](mpz_t r,mpz_srcptr x,mpz_srcptr y){mpz_mul(r,x,y);reduce(r);};
    auto sqr=[&](mpz_t r,mpz_srcptr x){mul(r,x,x);};
    auto inverse=[&](mpz_t r,mpz_srcptr x){return ecm_edwards::inverse(r,x,n,factor);};
    // Integer isomorphism of the auxiliary curve:
    // x=9s-3, y=27r gives y²=x³-756x+4320 and Q=(42,216).
    mpz_set_ui(Z.v,0);
    auto dbl=[&]() {
        if(!mpz_sgn(Z.v))return;
        sqr(A.v,X.v);sqr(B.v,Y.v);sqr(C.v,B.v);
        mul(S.v,X.v,B.v);mpz_mul_ui(S.v,S.v,4);reduce(S.v);
        sqr(M.v,Z.v);sqr(M.v,M.v);mpz_mul_ui(M.v,M.v,756);
        mpz_mul_ui(T.v,A.v,3);mpz_sub(M.v,T.v,M.v);reduce(M.v);
        sqr(U.v,M.v);mpz_sub(U.v,U.v,S.v);mpz_sub(U.v,U.v,S.v);reduce(U.v);
        mpz_sub(V.v,S.v,U.v);mul(V.v,M.v,V.v);mpz_mul_ui(T.v,C.v,8);
        mpz_sub(V.v,V.v,T.v);reduce(V.v);
        mul(Z.v,Y.v,Z.v);mpz_mul_ui(Z.v,Z.v,2);reduce(Z.v);
        mpz_set(X.v,U.v);mpz_set(Y.v,V.v);
    };
    for(int bit=63;bit>=0;--bit) {
        dbl();if(!((m>>bit)&1))continue;
        if(!mpz_sgn(Z.v)) {
            mpz_set_ui(X.v,42);reduce(X.v);mpz_set_ui(Y.v,216);reduce(Y.v);mpz_set_ui(Z.v,1);continue;
        }
        sqr(A.v,Z.v);mpz_mul_ui(U.v,A.v,42);reduce(U.v);
        mul(B.v,A.v,Z.v);mpz_mul_ui(V.v,B.v,216);reduce(V.v);
        mpz_sub(H.v,U.v,X.v);reduce(H.v);mpz_sub(J.v,V.v,Y.v);reduce(J.v);
        if(!mpz_sgn(H.v)) {
            if(!mpz_sgn(J.v)){dbl();continue;}
            mpz_gcd(factor,J.v,n);
            if(mpz_cmp_ui(factor,1)>0 && mpz_cmp(factor,n)<0)return Setup::Factor;
            mpz_set_ui(Z.v,0);continue;
        }
        sqr(A.v,H.v);mul(B.v,A.v,H.v);mul(K.v,X.v,A.v);
        sqr(U.v,J.v);mpz_sub(U.v,U.v,B.v);mpz_sub(U.v,U.v,K.v);mpz_sub(U.v,U.v,K.v);reduce(U.v);
        mpz_sub(V.v,K.v,U.v);mul(V.v,J.v,V.v);mul(T.v,Y.v,B.v);mpz_sub(V.v,V.v,T.v);reduce(V.v);
        mul(Z.v,Z.v,H.v);mpz_set(X.v,U.v);mpz_set(Y.v,V.v);
    }
    Setup rc=inverse(inv.v,Z.v);if(rc!=Setup::Ready)return rc;
    sqr(T.v,inv.v);mul(X.v,X.v,T.v);mul(T.v,T.v,inv.v);mul(Y.v,Y.v,T.v);
    // t=(4s+4)/(r-4)=(12x+144)/(y-108) in the integer model.
    mpz_mul_ui(num.v,X.v,12);mpz_add_ui(num.v,num.v,144);
    mpz_sub_ui(den.v,Y.v,108);rc=inverse(inv.v,den.v);if(rc!=Setup::Ready)return rc;
    mul(t.v,num.v,inv.v);
    auto poly=[&](mpz_t out,std::initializer_list<int> coefficients) {
        mpz_set_ui(out,0);
        for(int c:coefficients) {
            mul(out,out,t.v);
            if(c>=0)mpz_add_ui(out,out,unsigned(c));else mpz_sub_ui(out,out,unsigned(-c));
            reduce(out);
        }
    };
    poly(num.v,{1,4,0});poly(den.v,{1,0,-4});
    rc=inverse(inv.v,den.v);if(rc!=Setup::Ready)return rc;
    mul(e.v,num.v,inv.v);sqr(d,e.v);sqr(d,d);mpz_neg(d,d);reduce(d);
    // Nonsingular a=-1 requires d*(1+d) to be a unit; a24=1/(1+d).
    mpz_add_ui(den.v,d,1);mul(T.v,d,den.v);mpz_gcd(factor,T.v,n);
    if(mpz_cmp_ui(factor,1)!=0)return mpz_cmp(factor,n)<0?Setup::Factor:Setup::Degenerate;
    rc=inverse(a24,den.v);if(rc!=Setup::Ready)return rc;
    poly(num.v,{2,2,-8,-8});poly(den.v,{1,6,12,16,0});
    rc=inverse(inv.v,den.v);if(rc!=Setup::Ready)return rc;mul(px,num.v,inv.v);
    poly(num.v,{1,6,10,-16,-48,-32,-32});poly(den.v,{1,6,10,16,48,64,0});
    rc=inverse(inv.v,den.v);if(rc!=Setup::Ready)return rc;mul(py,num.v,inv.v);
    sqr(U.v,px);sqr(V.v,py);mpz_sub(T.v,V.v,U.v);mpz_sub_ui(T.v,T.v,1);
    mul(U.v,U.v,V.v);mul(U.v,U.v,d);mpz_sub(T.v,T.v,U.v);reduce(T.v);
    if(mpz_sgn(T.v))return Setup::Degenerate;
    mpz_set_ui(factor,0);return Setup::Ready;
}
}
