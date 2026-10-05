#define NTT_POLY_PROBE_NO_MAIN
#include "../bench/ntt_poly_probe.cu"
// Query the actual immutable production packing planner. No transforms or
// arithmetic kernels are launched; the device is selected explicitly.
int main(int argc,char **argv) {
    const int device=argc>1 ? std::atoi(argv[1]) : 1;
    CK(cudaSetDevice(device));
    unsigned long long p=0;int bits=0;
    while(std::scanf("%llu %d",&p,&bits)==2) {
        unsigned long long nf=0,nt=0;
        if(!p || bits<2 || bits>8192 || !ntt_shape_query(p+1,bits,&nf,nullptr,nullptr,nullptr,nullptr,nullptr) ||
           !ntt_shape_query(p/2+1,bits,&nt,nullptr,nullptr,nullptr,nullptr,nullptr))return 2;
        const auto big=24*nf,arena=8*(3*nf+2*p+1+2*(3*nt+2*(p/2+1)-1));
        std::printf("budget_shape: P=%llu bits=%d n_fold=%llu n_tree=%llu big_bytes=%llu arena_est_bytes=%llu\n",p,bits,nf,nt,big,arena);
    }
    return 0;
}
