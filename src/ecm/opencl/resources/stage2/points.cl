// Runtime nw bounds the loops; W bounds private capacity. Keeping loop bounds
// dynamic avoids quadratic JIT expansion, as in the CUDA capacity/nw contract.
// Native OpenCL C 1.2. SOS Montgomery multiplication follows the CUDA point
// contract; no subgroup assumptions, floating point, atomics, or vendor asm.
#define W ECM_POINT_NW
typedef ulong word;
void cp(int nw,word*r,const word*a){for(int i=0;i<nw;++i)r[i]=a[i];}
void zero(int nw,word*r){for(int i=0;i<nw;++i)r[i]=0;}
int nonzero(int nw,const word*a){word x=0;for(int i=0;i<nw;++i)x|=a[i];return x!=0;}
int one(int nw,const word*a){word x=a[0]^1UL;for(int i=1;i<nw;++i)x|=a[i];return x==0;}
int ge(int nw,const word*a,const word*b){for(int i=nw-1;i>=0;--i)if(a[i]!=b[i])return a[i]>b[i];return 1;}
void sub(int nw,word*r,const word*a,const word*b){word c=0;for(int i=0;i<nw;++i){word x=a[i]-b[i],b1=a[i]<b[i],y=x-c,b2=x<c;r[i]=y;c=b1|b2;}}
void addmod(int nw,word*r,const word*a,const word*b,const word*n){word c=0;for(int i=0;i<nw;++i){word x=a[i]+b[i],c1=x<a[i],y=x+c,c2=y<c;r[i]=y;c=c1|c2;}if(c||ge(nw,r,n))sub(nw,r,r,n);}
void submod(int nw,word*r,const word*a,const word*b,const word*n){int borrow=!ge(nw,a,b);sub(nw,r,a,b);if(borrow){word c=0;for(int i=0;i<nw;++i){word x=r[i]+n[i],c1=x<r[i],y=x+c,c2=y<c;r[i]=y;c=c1|c2;}}}
void shift(int nw,word*r,word carry){for(int i=nw-1;i>=0;--i){word low=r[i]&1UL;r[i]=(r[i]>>1)|(carry<<63);carry=low;}}
void halfmod(int nw,word*r,const word*n){word c=0;if(r[0]&1UL)for(int i=0;i<nw;++i){word x=r[i]+n[i],c1=x<r[i],y=x+c,c2=y<c;r[i]=y;c=c1|c2;}shift(nw,r,c);}
void mac(word x,word y,word z,word c,word*lo,word*hi){word p=x*y,h=mul_hi(x,y),s=p+z,k=s<p,t=s+c;k+=t<c;*lo=t;*hi=h+k;}
// Keep one SOS body instead of duplicating its private arrays at every xADD,
// xDBL and normalization call site (same compiler hint as special_mult_generic).
__attribute__((noinline))
void mont(int nw,word*r,const word*a,const word*b,const word*n,word ni){
    word t[2*W+2];for(int i=0;i<2*nw+2;++i)t[i]=0;
    // Unrolling nested fixed-width loops at every ladder call site makes JIT
    // grow rapidly. These are optimization hints; math does not depend on them.
    #pragma unroll 1
    for(int i=0;i<nw;++i){word c=0;
        #pragma unroll 1
        for(int j=0;j<nw;++j)mac(a[j],b[i],t[i+j],c,&t[i+j],&c);t[i+nw]=c;}
    #pragma unroll 1
    for(int i=0;i<nw;++i){word m=t[i]*ni,c=0;
        #pragma unroll 1
        for(int j=0;j<nw;++j)mac(m,n[j],t[i+j],c,&t[i+j],&c);
        int k=i+nw;word s=t[k]+c;c=s<t[k];t[k]=s;while(c){++k;s=t[k]+c;c=s<t[k];t[k]=s;}}
    for(int i=0;i<nw;++i)r[i]=t[nw+i];if(t[2*nw]||ge(nw,r,n))sub(nw,r,r,n);
}
void dbl(int nw,word*rx,word*rz,const word*x,const word*z,const word*a24,const word*n,word ni){
    word a[W],b[W],e[W],tx[W],tz[W];addmod(nw,a,x,z,n);submod(nw,b,x,z,n);mont(nw,a,a,a,n,ni);mont(nw,b,b,b,n,ni);
    submod(nw,e,a,b,n);mont(nw,tx,a,b,n,ni);mont(nw,tz,a24,e,n,ni);addmod(nw,tz,b,tz,n);mont(nw,tz,e,tz,n,ni);cp(nw,rx,tx);cp(nw,rz,tz);
}
void xadd(int nw,word*rx,word*rz,const word*x,const word*z,const word*y,const word*w,const word*dx,const word*dz,const word*n,word ni){
    // Half sums preserve the exact projective scaling of CUDA XADD6.
    word a[W],b[W],u[W],v[W],tx[W],tz[W];addmod(nw,a,x,z,n);submod(nw,b,y,w,n);mont(nw,u,a,b,n,ni);
    submod(nw,a,x,z,n);addmod(nw,b,y,w,n);mont(nw,v,a,b,n,ni);addmod(nw,tx,u,v,n);submod(nw,tz,u,v,n);halfmod(nw,tx,n);halfmod(nw,tz,n);
    mont(nw,tx,tx,tx,n,ni);mont(nw,tz,tz,tz,n,ni);mont(nw,tx,tx,dz,n,ni);mont(nw,tz,tz,dx,n,ni);cp(nw,rx,tx);cp(nw,rz,tz);
}
// Binary extended GCD: u == a*x (mod N), v == a*y (mod N).
// Both operands are nonnegative; every shift/subtraction reduces u or v.
// N must be odd. Return nonunit gcds, never treat inverse failure as zero.
__attribute__((noinline))
int inverse(int nw,word*inv,word*g,const word*a,const word*n,int extended){
    word u[W],v[W],x[W],y[W];cp(nw,u,a);cp(nw,v,n);zero(nw,x);zero(nw,y);x[0]=1;
    uint steps=0;
    while(nonzero(nw,u)&&nonzero(nw,v)){
        while(nonzero(nw,u)&&!(u[0]&1UL)){shift(nw,u,0);if(extended)halfmod(nw,x,n);if(++steps>8*64*nw+16)return 0;}
        while(nonzero(nw,v)&&!(v[0]&1UL)){shift(nw,v,0);if(extended)halfmod(nw,y,n);if(++steps>8*64*nw+16)return 0;}
        if(ge(nw,u,v)){sub(nw,u,u,v);if(extended)submod(nw,x,x,y,n);}else{sub(nw,v,v,u);if(extended)submod(nw,y,y,x,n);}
        if(++steps>8*64*nw+16)return 0;
    }
    if(nonzero(nw,u)){cp(nw,g,u);cp(nw,inv,x);}else{cp(nw,g,v);cp(nw,inv,y);}return 1;
}
void load(int nw,word*r,__global const word*a){for(int i=0;i<nw;++i)r[i]=a[i];}
void store(int nw,__global word*r,const word*a){for(int i=0;i<nw;++i)r[i]=a[i];}
// One ladder supplies BOTH adjacent seeds. All values here are Montgomery
// images; the caller converts only the coordinates it actually publishes.
// Inline this small control wrapper, retaining the shared noinline SOS body.
__attribute__((always_inline))
void ladder_pair(int nw,word*ax,word*az,word*bx,word*bz,const word*x,const word*z,const word*a24,const word*n,word ni,const word*mone,word k){
    if(!k){cp(nw,ax,mone);zero(nw,az);cp(nw,bx,x);cp(nw,bz,z);return;}
    cp(nw,ax,x);cp(nw,az,z);dbl(nw,bx,bz,x,z,a24,n,ni);int top=63;while(!((k>>top)&1UL))--top;
    for(int bit=top-1;bit>=0;--bit){if((k>>bit)&1UL){xadd(nw,ax,az,ax,az,bx,bz,x,z,n,ni);dbl(nw,bx,bz,bx,bz,a24,n,ni);}else{xadd(nw,bx,bz,ax,az,bx,bz,x,z,n,ni);dbl(nw,ax,az,ax,az,a24,n,ni);}}
}
__kernel void point_ladder(int nw,__global const word*c,word ni,__global const word*ks,word count,__global word*ox,__global word*oz){
    word id=get_global_id(0);if(id>=count)return;word n[W],mone[W],r2[W],x[W],z[W],a24[W],ax[W],az[W],bx[W],bz[W],plain[W];
    load(nw,n,c);load(nw,mone,c+nw);load(nw,r2,c+2*nw);load(nw,x,c+3*nw);load(nw,z,c+4*nw);load(nw,a24,c+5*nw);
    mont(nw,x,x,r2,n,ni);mont(nw,z,z,r2,n,ni);mont(nw,a24,a24,r2,n,ni);word k=ks[id];
    ladder_pair(nw,ax,az,bx,bz,x,z,a24,n,ni,mone,k);
    zero(nw,plain);plain[0]=1;mont(nw,ax,ax,plain,n,ni);mont(nw,az,az,plain,n,ni);store(nw,ox+id*nw,ax);store(nw,oz+id*nw,az);
}
__kernel void point_chain(int nw,__global const word*c,word ni,word first,word count,word segment,__global word*ox,__global word*oz){
    word lo=get_global_id(0)*segment;if(lo>=count)return;word hi=lo+min(segment,count-lo),k=first+lo;
    word n[W],mone[W],r2[W],x[W],z[W],a24[W],ax[W],az[W],bx[W],bz[W],tx[W],tz[W],plain[W];
    load(nw,n,c);load(nw,mone,c+nw);load(nw,r2,c+2*nw);load(nw,x,c+3*nw);load(nw,z,c+4*nw);load(nw,a24,c+5*nw);
    mont(nw,x,x,r2,n,ni);mont(nw,z,z,r2,n,ni);mont(nw,a24,a24,r2,n,ni);zero(nw,plain);plain[0]=1;
    ladder_pair(nw,ax,az,bx,bz,x,z,a24,n,ni,mone,k);
    mont(nw,tx,ax,plain,n,ni);mont(nw,tz,az,plain,n,ni);store(nw,ox+lo*nw,tx);store(nw,oz+lo*nw,tz);
    if(lo+1<hi){mont(nw,tx,bx,plain,n,ni);mont(nw,tz,bz,plain,n,ni);store(nw,ox+(lo+1)*nw,tx);store(nw,oz+(lo+1)*nw,tz);}
    for(word j=lo+2;j<hi;++j){
        // [k+1]Q + Q has difference [k]Q, not Q. Starting at zero
        // needs a doubling because differential addition cannot double Q.
        if(k==0&&j==lo+2)dbl(nw,tx,tz,bx,bz,a24,n,ni);
        else xadd(nw,tx,tz,bx,bz,x,z,ax,az,n,ni);
        cp(nw,ax,bx);cp(nw,az,bz);cp(nw,bx,tx);cp(nw,bz,tz);
        mont(nw,tx,bx,plain,n,ni);mont(nw,tz,bz,plain,n,ni);store(nw,ox+j*nw,tx);store(nw,oz+j*nw,tz);
    }
}
__kernel void point_normalize(int nw,__global const word*c,word ni,__global const word*xs,__global const word*zs,word offset,word stride,word count,word segment,
    __global word*prefix,__global word*out,__global word*gs,__global word*bad){
    word lo=get_global_id(0)*segment;if(lo>=count)return;word hi=min(lo+segment,count),n[W],r2[W],acc[W],z[W],x[W],inv[W],g[W],prev[W],plain[W];
    load(nw,n,c);load(nw,r2,c+2*nw);load(nw,acc,c+nw);zero(nw,plain);plain[0]=1;
    for(word j=lo;j<hi;++j){load(nw,z,zs+offset+j*stride);mont(nw,z,z,r2,n,ni);mont(nw,acc,acc,z,n,ni);store(nw,prefix+j*nw,acc);}
    mont(nw,acc,acc,plain,n,ni);if(!inverse(nw,inv,g,acc,n,1)){bad[get_global_id(0)]=1;return;}
    if(one(nw,g)){
        mont(nw,inv,inv,r2,n,ni);
        for(word j=hi;j-->lo;){if(j==lo)load(nw,prev,c+nw);else load(nw,prev,prefix+(j-1)*nw);mont(nw,acc,inv,prev,n,ni);
            load(nw,x,xs+offset+j*stride);mont(nw,x,x,r2,n,ni);mont(nw,x,x,acc,n,ni);mont(nw,x,x,plain,n,ni);store(nw,out+j*nw,x);store(nw,gs+j*nw,plain);
            load(nw,z,zs+offset+j*stride);mont(nw,z,z,r2,n,ni);mont(nw,inv,inv,z,n,ni);}
        bad[get_global_id(0)]=0;
    }else bad[get_global_id(0)]=2; // Nonunit: handled by the recovery kernel.
}
__kernel void point_recover(int nw,__global const word*c,word ni,__global const word*xs,__global const word*zs,word offset,word stride,word count,word segment,
    __global word*out,__global word*gs,__global word*bad){
    word id=get_global_id(0);if(id>=count)return;word groups=(count+segment-1)/segment;bad[groups+id]=0;
    if(bad[id/segment]!=2)return;
    // Separate cold path keeps failed segments out of hot batch inversion.
    // Device qualification limits are enforced by the host before construction.
    word n[W],r2[W],z[W],x[W],inv[W],g[W];load(nw,n,c);load(nw,r2,c+2*nw);load(nw,z,zs+offset+id*stride);
    if(!inverse(nw,inv,g,z,n,1)){bad[groups+id]=1;return;}zero(nw,x);
    if(one(nw,g)){mont(nw,inv,inv,r2,n,ni);load(nw,x,xs+offset+id*stride);mont(nw,x,x,inv,n,ni);}store(nw,out+id*nw,x);store(nw,gs+id*nw,g);
}
__kernel void point_products(int nw,__global const word*c,word ni,__global const word*values,word offset,word stride,word count,word block,__global word*out){
    word lo=get_global_id(0)*block;if(lo>=count)return;word n[W],r2[W],acc[W],v[W],plain[W];load(nw,n,c);load(nw,r2,c+2*nw);load(nw,acc,c+nw);
    for(word j=lo;j<min(lo+block,count);++j){load(nw,v,values+offset+j*stride);mont(nw,v,v,r2,n,ni);mont(nw,acc,acc,v,n,ni);}zero(nw,plain);plain[0]=1;mont(nw,acc,acc,plain,n,ni);store(nw,out+get_global_id(0)*nw,acc);
}
__kernel void point_gcd(int nw,__global const word*c,__global const word*values,word offset,word stride,word count,__global word*out,__global word*bad){
    word id=get_global_id(0);if(id>=count)return;word n[W],a[W],g[W],unused[W];load(nw,n,c);load(nw,a,values+offset+id*stride);
    bad[id]=!inverse(nw,unused,g,a,n,0);store(nw,out+id*nw,g);
}
