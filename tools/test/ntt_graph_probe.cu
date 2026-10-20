#define NTT_POLY_PROBE_NO_MAIN
#include "../../src/core/ecm_stage2_logging.h"
#include "../../src/cuda/ntt/runtime.h"

using Word=unsigned long long;
__global__ void graph_corrupt(Word *digits){if(threadIdx.x==0 && blockIdx.x==0)digits[0]^=1;}
struct Readback {
    std::vector<Word> digits;Word n=0,slot_words=0;int bpw=0;
    bool inject=false,injected=false;
    static void run(void *ctx,const Word *digits,Word n,int bpw,Word sw,Word,
        Word,Word batch,Word *,Word,Word,Word) {
        auto &self=*(Readback *)ctx;self.n=n;self.bpw=bpw;self.slot_words=sw;
        if(self.inject && !self.injected){graph_corrupt<<<1,32>>>(const_cast<Word *>(digits));self.injected=true;}
        self.digits.resize((size_t)n*batch);
        CK(cudaMemcpy(self.digits.data(),digits,self.digits.size()*8,cudaMemcpyDeviceToHost));
    }
};
static Word next_random(Word &x){x^=x>>12;x^=x<<25;x^=x>>27;return x*2685821657736338717ull;}
int main(int argc,char **argv)
{
    const int device=argc>1?std::atoi(argv[1]):1;const bool fault=argc>2 && !std::strcmp(argv[2],"--fault");
    CK(cudaSetDevice(device));std::setvbuf(stdout,nullptr,_IONBF,0);stage2_log::level=stage2_log::debug;
    _putenv_s("NTT_SHORT_GRAPH_MIN_CALLS","2");_putenv_s("NTT_SHORT_GRAPH_MAX_WORDS","67108864");
    _putenv_s("NTT_SHORT_GRAPH_ENTRIES","4");_putenv_s("NTT_FUSE_WARP_TAIL","1");
    Word checks=0,bad=0,words=0,builds=0,hits=0,invalidations=0;bool injected=false;
    const unsigned cases[][3]={{3,129,1},{17,521,3},{65,2203,7},{65,7995,3},{257,521,9}};
    for(bool enabled:{false,true})for(bool reuse_bq:{false,true}) {
        _putenv_s("NTT_SHORT_GRAPH",enabled?"1":"0");NttArena ar;ar.device=device;ar.workspace_pool=true;ar.workspace_reuse_bq=reuse_bq;
        for(const auto &item:cases) {
            const Word p=item[0],bits=item[1],batch=item[2],w=(bits+63)/64;Word seed=0x5813;
            std::vector<Word> a((size_t)p*w*batch),b(a.size());
            for(size_t i=0;i<a.size();++i){a[i]=next_random(seed);b[i]=next_random(seed);}
            if(bits%64)for(size_t i=w-1;i<a.size();i+=w){a[i]&=(1ull<<(bits%64))-1;b[i]&=(1ull<<(bits%64))-1;}
            for(int repeat=0;repeat<4;++repeat) {
                Readback read;read.inject=fault && enabled && !injected && repeat==2;
                NttReduceHook hook;hook.ctx=&read;hook.run=Readback::run;hook.w=1;hook.count=0;
                NttMulStats stats;const int rc=ntt_poly_mul_batch_host(p,(int)bits,device,batch,a.data(),b.data(),nullptr,&stats,&ar,&hook);
                if(rc || read.digits.size()!=read.n*batch)return 2;
                if(read.injected)injected=true;
                mpz_t x,y,u,z,got;mpz_inits(x,y,u,z,got,nullptr);
                for(Word s=0;s<batch;++s) {
                    mpz_set_ui(x,0);mpz_set_ui(y,0);
                    for(Word i=p;i-- >0;) {
                        mpz_mul_2exp(x,x,(mp_bitcnt_t)stats.slot_stride);mpz_import(u,w,-1,8,0,0,&a[(s*p+i)*w]);mpz_add(x,x,u);
                        mpz_mul_2exp(y,y,(mp_bitcnt_t)stats.slot_stride);mpz_import(u,w,-1,8,0,0,&b[(s*p+i)*w]);mpz_add(y,y,u);
                    }
                    mpz_mul(z,x,y);mpz_import(got,read.n,-1,8,0,64-read.bpw,&read.digits[s*read.n]);
                    ++checks;if(mpz_cmp(z,got))++bad;
                    for(Word j=0;j<read.n;++j){++words;if(read.digits[s*read.n+j]>=(1ull<<read.bpw))++bad;}
                }
                mpz_clears(x,y,u,z,got,nullptr);
                std::printf("ntt_graph_case: enabled=%d reuse_bq=%d P=%llu S=%llu batch=%llu repeat=%d N=%llu builds=%llu hits=%llu bad=%llu injected=%d\n",
                    (int)enabled,(int)reuse_bq,p,bits,batch,repeat+1,stats.N,ar.short_graph.builds,ar.short_graph.hits,bad,(int)read.injected);
            }
            ar.evict_other_shapes(0,0); // tables and non-pool storage invalidation
        }
        builds+=ar.short_graph.builds;hits+=ar.short_graph.hits;invalidations+=ar.short_graph.invalidations;
        ar.release();if(ar.bytes || !ar.short_graph.entries.empty() || ar.short_graph.capture_stream)++bad;
    }
    _putenv_s("NTT_SHORT_GRAPH","1");
    if(!fault){ntt_workspace_check(device);ntt_fuse_lifetime_check(device);ntt_fuse_capacity_check(device);}
    if(!builds || !hits || !invalidations)++bad;
    std::printf("ntt_graph_gate: checks=%llu checked_words=%llu builds=%llu hits=%llu invalidations=%llu fault=%d bad=%llu fuse_live_bytes=%llu\n",
        checks,words,builds,hits,invalidations,(int)fault,bad,g_fuse_base.live_bytes);
    return bad?3:0;
}
