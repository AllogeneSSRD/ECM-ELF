// Independent x-only ladder verifies the complete Edwards 48*lcm saved point.
// Stage1's production scalar multiplication uses Edwards point formulas.
#include "ecm_edwards_save_reader.h"
#include <iostream>
#include <stdexcept>
using namespace ecm_edwards;
int main(int argc,char **argv){try{
    if(argc!=2)throw std::runtime_error("expected one text save");
    for(const auto &row:read_saved_curves(argv[1])){
        if(row.factor)throw std::runtime_error("POINT required");
        if(row.b1>1000000)throw std::runtime_error("verification bound exceeds benchmark scope");
        Integer n,d,px,py,a24,f,scalar,x,z,rx,rz,sx,sz,tx,tz,aa,bb,cc,dd,u,v;
        mpz_set_str(n.v,row.n.c_str(),16);
        if(construct(d.v,px.v,py.v,a24.v,f.v,row.sigma,n.v)!=Setup::Ready)throw std::runtime_error("base point setup failed");
        mpz_add_ui(x.v,py.v,1);mpz_ui_sub(z.v,1,py.v);
        if(!mpz_invert(z.v,z.v,n.v))throw std::runtime_error("base point denominator is not invertible");
        mpz_mul(x.v,x.v,z.v);mpz_mod(x.v,x.v,n.v);
        mpz_set_ui(scalar.v,1);for(unsigned long k=2;k<=row.b1;++k)mpz_lcm_ui(scalar.v,scalar.v,k);
        mpz_mul_ui(scalar.v,scalar.v,48);
        auto dbl=[&](mpz_t ox,mpz_t oz,const mpz_t ax,const mpz_t az){
            mpz_add(aa.v,ax,az);mpz_mul(aa.v,aa.v,aa.v);mpz_mod(aa.v,aa.v,n.v);
            mpz_sub(bb.v,ax,az);mpz_mul(bb.v,bb.v,bb.v);mpz_mod(bb.v,bb.v,n.v);
            mpz_sub(cc.v,aa.v,bb.v);mpz_mod(cc.v,cc.v,n.v);
            mpz_mul(ox,aa.v,bb.v);mpz_mod(ox,ox,n.v);
            mpz_mul(u.v,a24.v,cc.v);mpz_add(u.v,u.v,bb.v);mpz_mul(oz,u.v,cc.v);mpz_mod(oz,oz,n.v);
        };
        auto add=[&](mpz_t ox,mpz_t oz,const mpz_t ax,const mpz_t az,const mpz_t bx,const mpz_t bz){
            mpz_add(aa.v,ax,az);mpz_sub(bb.v,ax,az);mpz_add(cc.v,bx,bz);mpz_sub(dd.v,bx,bz);
            mpz_mul(u.v,aa.v,dd.v);mpz_mod(u.v,u.v,n.v);mpz_mul(v.v,bb.v,cc.v);mpz_mod(v.v,v.v,n.v);
            mpz_add(ox,u.v,v.v);mpz_mul(ox,ox,ox);mpz_mod(ox,ox,n.v);
            mpz_sub(oz,u.v,v.v);mpz_mul(oz,oz,oz);mpz_mod(oz,oz,n.v);mpz_mul(oz,oz,x.v);mpz_mod(oz,oz,n.v);
        };
        mpz_set(rx.v,x.v);mpz_set_ui(rz.v,1);dbl(sx.v,sz.v,rx.v,rz.v);
        for(auto bit=mpz_sizeinbase(scalar.v,2)-1;bit-->0;){
            add(tx.v,tz.v,rx.v,rz.v,sx.v,sz.v);
            if(mpz_tstbit(scalar.v,bit)){dbl(sx.v,sz.v,sx.v,sz.v);mpz_set(rx.v,tx.v);mpz_set(rz.v,tz.v);}
            else{dbl(rx.v,rz.v,rx.v,rz.v);mpz_set(sx.v,tx.v);mpz_set(sz.v,tz.v);}
        }
        mpz_gcd(f.v,rz.v,n.v);if(mpz_cmp_ui(f.v,1))throw std::runtime_error("ladder denominator is not invertible");
        mpz_set_str(tx.v,row.x.c_str(),16);mpz_mul(tx.v,tx.v,rz.v);mpz_sub(tx.v,tx.v,rx.v);mpz_mod(tx.v,tx.v,n.v);
        if(mpz_sgn(tx.v))throw std::runtime_error("independent ladder differs from saved point");
        std::cout<<"verified bits="<<mpz_sizeinbase(n.v,2)<<" B1="<<row.b1<<" sigma="<<row.sigma<<'\n';
    }
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
