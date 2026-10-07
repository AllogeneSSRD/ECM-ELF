/* One [D]Q on the CPU. Inputs/outputs are Montgomery images, arithmetic is
   normal-domain GMP. Preserve the GPU ladder's exact projective scale; no
   affine normalization or inverse of a point coordinate is performed. */
static bool stage2_giant_base_gmp(size_t nw,unsigned long long d,mpz_srcptr N,
                                 const unsigned long long *qxm,const unsigned long long *qzm,
                                 const unsigned long long *a24m,const unsigned long long *rm,
                                 std::vector<unsigned long long> &out)
{
    mpz_t x,z,a24,R,ri,ax,az,bx,bz,t1,t2,t3,t4,tx,tz,gcd;
    mpz_inits(x,z,a24,R,ri,ax,az,bx,bz,t1,t2,t3,t4,tx,tz,gcd,nullptr);
    words_to_mpz(R,rm,nw);
    if(!mpz_invert(ri,R,N)) {
        std::fprintf(stderr,"FATAL: CPU giant base requires invertible Montgomery radix\n");
        std::exit(3);
    }
    auto decode=[&](mpz_ptr v,const unsigned long long *image) {
        words_to_mpz(v,image,nw);mpz_mul(v,v,ri);mpz_mod(v,v,N);
    };
    decode(x,qxm);decode(z,qzm);decode(a24,a24m);
    auto dbl=[&](mpz_ptr rx,mpz_ptr rz,mpz_srcptr px,mpz_srcptr pz) {
        mpz_add(t1,px,pz);mpz_mul(t3,t1,t1);mpz_mod(t3,t3,N);
        mpz_sub(t2,px,pz);mpz_mul(t4,t2,t2);mpz_mod(t4,t4,N);
        mpz_mul(tx,t3,t4);mpz_mod(tx,tx,N);
        mpz_sub(t1,t3,t4);mpz_mod(t1,t1,N);
        mpz_mul(t2,a24,t1);mpz_add(t2,t2,t4);mpz_mod(t2,t2,N);
        mpz_mul(tz,t1,t2);mpz_mod(tz,tz,N);
        mpz_set(rx,tx);mpz_set(rz,tz); // Both writes after all input reads.
    };
    auto add=[&](mpz_ptr rx,mpz_ptr rz,mpz_srcptr px,mpz_srcptr pz,
                 mpz_srcptr qx,mpz_srcptr qz) {
        // The original eight-multiply formula has exactly the xADD6 scale.
        mpz_mul(t1,px,qx);mpz_mul(t2,pz,qz);mpz_sub(t1,t1,t2);mpz_mod(t1,t1,N);
        mpz_mul(t1,t1,t1);mpz_mod(t1,t1,N);mpz_mul(tx,z,t1);mpz_mod(tx,tx,N);
        mpz_mul(t1,px,qz);mpz_mul(t2,pz,qx);mpz_sub(t1,t1,t2);mpz_mod(t1,t1,N);
        mpz_mul(t1,t1,t1);mpz_mod(t1,t1,N);mpz_mul(tz,x,t1);mpz_mod(tz,tz,N);
        mpz_set(rx,tx);mpz_set(rz,tz);
    };
    if(!d) {mpz_set_ui(ax,1);mpz_set_ui(az,0);}
    else {
        mpz_set(ax,x);mpz_set(az,z);dbl(bx,bz,x,z);
        int top=63;while(((d>>top)&1ull)==0)--top;
        for(int bit=top-1;bit>=0;--bit) {
            if((d>>bit)&1ull) {add(ax,az,ax,az,bx,bz);dbl(bx,bz,bx,bz);}
            else {add(bx,bz,ax,az,bx,bz);dbl(ax,az,ax,az);}
        }
    }
    mpz_gcd(gcd,az,N);const bool unit=mpz_cmp_ui(gcd,1)==0;
    out.assign(2*nw,0ull);
    std::vector<unsigned long long> word;
    mpz_mul(ax,ax,R);mpz_mod(ax,ax,N);mpz_to_words(word,nw,ax);
    std::copy(word.begin(),word.end(),out.begin());
    mpz_mul(az,az,R);mpz_mod(az,az,N);mpz_to_words(word,nw,az);
    std::copy(word.begin(),word.end(),out.begin()+nw);
    mpz_clears(x,z,a24,R,ri,ax,az,bx,bz,t1,t2,t3,t4,tx,tz,gcd,nullptr);
    return unit;
}
