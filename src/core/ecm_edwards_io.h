#pragma once
#include "ecm_edwards_curve.h"
#include <string>
#include <vector>

namespace ecm_edwards {
inline std::string hex(const mpz_t x) {
    std::vector<char> buf(mpz_sizeinbase(x,16)+3);mpz_get_str(buf.data(),16,x);return buf.data();
}
inline uint64_t hash(const std::string &s) {
    uint64_t h=14695981039346656037ull;
    for(unsigned char ch:s){h^=ch;h*=1099511628211ull;}return h;
}
inline uint64_t save_checksum(const std::string &n,const std::string &x,
                               uint64_t sigma,uint64_t b1,bool factor) {
    return hash("ECM-EDWARDS-AM|1|48LCM|"+n+"|"+x+"|"+std::to_string(sigma)+"|"+
                std::to_string(b1)+"|"+(factor?"FACTOR":"POINT"));
}
inline std::string save_line(const mpz_t n,const mpz_t x,uint64_t sigma,uint64_t b1,bool factor) {
    const auto nh=hex(n),xh=hex(x);
    return "METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA="+std::to_string(sigma)+
        "; B1="+std::to_string(b1)+"; N=0x"+nh+"; X=0x"+xh+"; STATUS="+
        (factor?"FACTOR":"POINT")+"; CHECKSUM="+std::to_string(save_checksum(nh,xh,sigma,b1,factor))+";\n";
}
// Validate raw extended coordinates; reducing corrupt input would hide errors.
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
