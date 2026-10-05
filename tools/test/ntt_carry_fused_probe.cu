#define NTT_POLY_PROBE_NO_MAIN
#include "../bench/ntt_poly_probe.cu"
using U=unsigned long long;
__global__ void sparse_pack(U *a,U *b,U n,U p,U slot_words){const auto i=blockIdx.x*256ull+threadIdx.x;const auto s=blockIdx.y;if(i<n){a[(U)s*n+i]=i==0 ? 2+s : i==(p-1)*slot_words ? 3+s : 0;b[(U)s*n+i]=i==0 ? 5+s : i==(p-1)*slot_words ? 7+s : 0;}}
static int input(void *,const NttShape &sh,U batch,U *a,U *b){sparse_pack<<<dim3((unsigned int)((sh.N+255)/256),(unsigned int)batch),256>>>(a,b,sh.N,sh.P,sh.slot_words);CK(cudaGetLastError());return 0;}
int main(int argc,char **argv){
    const int device=argc>1 ? std::atoi(argv[1]) : 1;CK(cudaSetDevice(device));
    std::printf("carry_fused_device: %d\n",device);U cases=0,bad=0,words=0;
    NttArena arena;arena.device=device;arena.cap_bytes=512ull<<20;NttInputHook hook{nullptr,input};
    const U p=2048,batch=3;const int bits=4423;
    auto multiply=[&](bool defer){NttMulStats st{};const int rc=ntt_poly_mul_batch_dev(p,bits,device,batch,nullptr,nullptr,&st,&arena,nullptr,nullptr,0,defer,&hook);
        ++cases;bad+=rc!=0;bad+=st.carry_deferred!=defer;
        if(rc)return;
        std::vector<U> out((2*p-1)*batch);CK(cudaMemcpy(out.data(),arena.cur.dOut,out.size()*8,cudaMemcpyDeviceToHost));
        for(U s=0;s<batch;++s)for(U i=0;i<2*p-1;++i){const U want=i==0 ? (2+s)*(5+s) : i==p-1 ? (2+s)*(7+s)+(3+s)*(5+s) : i==2*p-2 ? (3+s)*(7+s) : 0;bad+=out[s*(2*p-1)+i]!=want;++words;}
    };
    fuse_fixture_env("NTT_CARRY_CHECK_FUSED","0");multiply(false);bad+=arena.carry_fused_calls!=0 || arena.carry_scratch_bytes!=0;
    fuse_fixture_env("NTT_CARRY_CHECK_FUSED","1");multiply(false);bad+=arena.carry_fused_calls!=1 || arena.carry_scratch_bytes==0;
    const U n=arena.cur.n;const auto scratch=arena.carry_scratch;const auto grows=arena.carry_grows;
    CK(cudaMemset(arena.cur.dRes,0,batch*16));multiply(true);multiply(true);multiply(true);
    NttMulStats finish{};++cases;bad+=ntt_batch_carry_finish(&arena,n,batch,&finish)!=0 || finish.carry_residual!=0;
    bad+=arena.carry_scratch!=scratch || arena.carry_grows!=grows;
    // A previous interior chunk's nonzero count survives all later good kernels.
    U poison=9;CK(cudaMemcpy(arena.cur.dRes,&poison,8,cudaMemcpyHostToDevice));multiply(true);
    ++cases;bad+=ntt_batch_carry_finish(&arena,n,batch,&finish)!=4 || finish.carry_residual!=9;
    multiply(false);++cases;bad+=ntt_batch_carry_finish(&arena,n,batch,&finish)!=0;
    fuse_fixture_env("NTT_CARRY_CHECK_ALLOC_FAIL","1");auto used=arena.carry_fused_calls;auto refused=arena.carry_refusals;
    multiply(false);bad+=arena.carry_fused_calls!=used || arena.carry_refusals<=refused;
    fuse_fixture_env("NTT_CARRY_CHECK_ALLOC_FAIL","");arena.drop_carry_scratch();const auto original_cap=arena.cap_bytes;arena.cap_bytes=arena.bytes;
    refused=arena.carry_refusals;multiply(false);bad+=arena.carry_scratch_bytes!=0 || arena.carry_refusals<=refused;
    arena.cap_bytes=original_cap;multiply(false);bad+=arena.carry_scratch_bytes==0;
    const auto skipped=arena.carry_skipped_calls;++cases;bad+=arena.carry_workspace(4096,1)!=nullptr || arena.carry_skipped_calls!=skipped+1;
    ++cases;bad+=arena.carry_workspace(~0ull,2)!=nullptr;
    arena.print_workspace_stats();arena.release();++cases;bad+=arena.bytes!=0 || arena.carry_scratch!=nullptr || arena.carry_scratch_bytes!=0;
    std::printf("carry_fused_gate: cases=%llu words=%llu bad=%llu (actual NTT runner, cached switches, deferred fault, reset, cap/allocator fallback, small/overflow, release)\n",cases,words,bad);
    return bad ? 3 : 0;
}
