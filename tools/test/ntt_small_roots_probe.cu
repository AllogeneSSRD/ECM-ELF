#define NTT_TENSOR_GOLDILOCKS_PROBE_NO_MAIN
#include "ntt_tensor_goldilocks_probe.cu" // reuse independent GMP tile oracle and pattern buffers
#include "../bench/ntt_small_roots.cuh"

template<int METHOD>
__global__ void root_primitive(const Word *v,const Word *w,const unsigned int *j,
    const unsigned int *st,const unsigned int *inv,Word *out,Word count)
{
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    if constexpr(METHOD==1)out[i]=small_root_value(v[i],w[i],st[i]);
    else out[i]=inv[i] ? small_root_index<true,METHOD==3>(v[i],j[i],st[i]) : small_root_index<false,METHOD==3>(v[i],j[i],st[i]);
}
static void root_attributes(int shared)
{
    const void *kernels[]={(const void *)small_root_tile<1,false>,(const void *)small_root_tile<1,true>,
                          (const void *)small_root_tile<2,false>,(const void *)small_root_tile<2,true>,
                          (const void *)small_root_tile<3,false>,(const void *)small_root_tile<3,true>};
    for(const auto kernel:kernels) {
        CK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
        CK(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,100));
    }
    ntt_tile_attributes(true,shared,true);
}
static void root_launch(int method,bool inverse,dim3 grid,int t,Word *a,const Word *b,
    const Word *table,Word scale,Word stride)
{
    const Word n=(Word)grid.x*(1ull<<t);const unsigned int shared=(1u<<t)*8;
    if(method==0) {
        if(inverse)tile_kernel<true,true><<<grid,512,shared>>>(a,b,n,t,t,table,scale,stride);
        else tile_kernel<false,true><<<grid,512,shared>>>(a,nullptr,n,t,t,table,0,stride);
    } else if(method==1) {
        if(inverse)small_root_tile<1,true><<<grid,512,shared>>>(a,b,n,t,table,scale,stride);
        else small_root_tile<1,false><<<grid,512,shared>>>(a,nullptr,n,t,table,0,stride);
    } else if(method==2) {
        if(inverse)small_root_tile<2,true><<<grid,512,shared>>>(a,b,n,t,table,scale,stride);
        else small_root_tile<2,false><<<grid,512,shared>>>(a,nullptr,n,t,table,0,stride);
    } else {
        if(inverse)small_root_tile<3,true><<<grid,512,shared>>>(a,b,n,t,table,scale,stride);
        else small_root_tile<3,false><<<grid,512,shared>>>(a,nullptr,n,t,table,0,stride);
    }
    CK(cudaGetLastError());
}
static int primitive_check(bool fault)
{
    std::vector<Word> v,w,want;std::vector<unsigned int> j,st,inv;
    auto values=fixture(5,4096,0x3671);const Word edges[]={0,1,2,GL_P-1,GL_P-2,0xffffffffull,1ull<<32,1ull<<63,GL_P/2,0x0001000100010001ull};
    values.insert(values.end(),std::begin(edges),std::end(edges));
    mpz_t p,x,y,z;mpz_inits(p,x,y,z,nullptr);put64(p,GL_P);
    auto add=[&](Word value,Word root,unsigned int index,unsigned int stage,unsigned int inverse) {
        put64(x,value);put64(y,root);mpz_mul(z,x,y);mpz_mod(z,z,p);
        v.push_back(value);w.push_back(root);want.push_back(get64(z));j.push_back(index);st.push_back(stage);inv.push_back(inverse);
    };
    for(unsigned int inverse:{0u,1u}) {
        Transform transform(inverse,false);
        if(!inverse && transform.matrix[17]!=GL_P-(1ull<<60))std::exit(2);
        for(unsigned int stage:{2u,3u})for(unsigned int index=0;index<(1u<<stage);++index) {
            const auto root=transform.matrix[16+index*(1u<<(3-stage))];
            for(auto value:values)add(value,root,index,stage,inverse);
        }
    }
    Buffer dv(v.size()*8),dw(w.size()*8),dj(j.size()*4),ds(st.size()*4),di(inv.size()*4),out(v.size()*8);
    CK(cudaMemcpy(dv.p,v.data(),dv.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(dw.p,w.data(),dw.bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dj.p,j.data(),dj.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds.p,st.data(),ds.bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(di.p,inv.data(),di.bytes,cudaMemcpyHostToDevice));
    std::vector<Word> got(v.size());Word bad=0;
    for(int method:{1,2,3}) {
        const auto grid=(unsigned int)((v.size()+255)/256);
        if(method==1)root_primitive<1><<<grid,256>>>(dv.get(),dw.get(),dj.get<unsigned int>(),ds.get<unsigned int>(),di.get<unsigned int>(),out.get(),v.size());
        else if(method==2)root_primitive<2><<<grid,256>>>(dv.get(),dw.get(),dj.get<unsigned int>(),ds.get<unsigned int>(),di.get<unsigned int>(),out.get(),v.size());
        else root_primitive<3><<<grid,256>>>(dv.get(),dw.get(),dj.get<unsigned int>(),ds.get<unsigned int>(),di.get<unsigned int>(),out.get(),v.size());
        if(fault && method==1)corrupt_first<<<1,32>>>(out.get());
        CK(cudaMemcpy(got.data(),out.p,out.bytes,cudaMemcpyDeviceToHost));Checks result;compare(result,got,want);bad+=result.bad;
        std::printf("small_root_check: method=%d group=primitive cases=%llu words=%llu bad=%llu\n",method,result.cases,result.words,result.bad);
    }
    v.clear();w.clear();want.clear();j.clear();st.clear();inv.clear();
    const auto other=fixture(5,32,0x9e13);
    for(unsigned int stage:{2u,3u})for(auto root:other)for(size_t index=0;index<32;++index)add(values[index],root,0,stage,0);
    Buffer fv(v.size()*8),fw(w.size()*8),fs(st.size()*4),fj(j.size()*4),fi(inv.size()*4),fo(v.size()*8);
    CK(cudaMemcpy(fv.p,v.data(),fv.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(fw.p,w.data(),fw.bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fs.p,st.data(),fs.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(fj.p,j.data(),fj.bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fi.p,inv.data(),fi.bytes,cudaMemcpyHostToDevice));
    root_primitive<1><<<(unsigned int)((v.size()+255)/256),256>>>(fv.get(),fw.get(),fj.get<unsigned int>(),fs.get<unsigned int>(),fi.get<unsigned int>(),fo.get(),v.size());
    got.resize(v.size());CK(cudaMemcpy(got.data(),fo.p,fo.bytes,cudaMemcpyDeviceToHost));Checks fallback;compare(fallback,got,want);bad+=fallback.bad;
    std::printf("small_root_check: method=1 group=fallback cases=%llu words=%llu bad=%llu\n",fallback.cases,fallback.words,fallback.bad);
    mpz_clears(p,x,y,z,nullptr);std::printf("small_root_primitive_gate: fault=%d bad=%llu\n",(int)fault,bad);return bad ? 3 : 0;
}
__global__ void root_reduce_kernel(const Word *lo,const Word *hi,Word *out,Word count)
{
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=small_root_reduce(lo[i],hi[i]);
}
static int reduce_check()
{
    std::vector<Word> lo,hi,want;Word seed=0x33913;
    const Word edges[]={0,1,GL_P-1,GL_P,~0ull,0xffffffffull,1ull<<32,1ull<<63};
    for(auto a:edges)for(auto b:edges){lo.push_back(a);hi.push_back(b);}
    for(int i=0;i<100000;++i){lo.push_back(random64(seed));hi.push_back(random64(seed));}
    mpz_t p,x,z;mpz_inits(p,x,z,nullptr);put64(p,GL_P);
    for(size_t i=0;i<lo.size();++i){put64(z,hi[i]);mpz_mul_2exp(z,z,64);put64(x,lo[i]);mpz_add(z,z,x);mpz_mod(z,z,p);want.push_back(get64(z));}
    mpz_clears(p,x,z,nullptr);Buffer dl(lo.size()*8),dh(hi.size()*8),out(lo.size()*8);
    CK(cudaMemcpy(dl.p,lo.data(),dl.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(dh.p,hi.data(),dh.bytes,cudaMemcpyHostToDevice));
    root_reduce_kernel<<<(unsigned int)((lo.size()+255)/256),256>>>(dl.get(),dh.get(),out.get(),lo.size());
    std::vector<Word> got(lo.size());CK(cudaMemcpy(got.data(),out.p,out.bytes,cudaMemcpyDeviceToHost));Checks result;compare(result,got,want);
    std::printf("small_root_check: method=3 group=reduce128 cases=%llu words=%llu bad=%llu\n",result.cases,result.words,result.bad);return result.bad ? 3 : 0;
}
static int tile_check()
{
    Checks forward[4],inverse[4],roundtrip[4],readonly[4],guard[4];
    for(int t:{5,6,8,10,12}) {
        const unsigned int tile=1u<<t;TilePlan plan(t);root_attributes(tile*8);
        Buffer tf(plan.forward.size()*8),ti(plan.inverse.size()*8);
        CK(cudaMemcpy(tf.p,plan.forward.data(),tf.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(ti.p,plan.inverse.data(),ti.bytes,cudaMemcpyHostToDevice));
        for(int type=0;type<8;++type)for(unsigned int batches:{1u,3u})for(unsigned int tiles:{1u,3u}) {
            const unsigned int padding=17,stride=tile*tiles+padding;const dim3 grid(tiles,batches);
            auto input=fixture(type,(size_t)stride*batches,0x354),multiplier=fixture((type+3)%8,input.size(),0x876);
            const auto fw=tile_reference(input,multiplier,t,false,tiles,batches,padding),iv=tile_reference(input,multiplier,t,true,tiles,batches,padding);
            auto ones=std::vector<Word>(input.size(),1);std::vector<Word> got(input.size()),bgot(input.size());
            Buffer a(input.size()*8),b(input.size()*8);
            for(int method:{0,1,2,3}) {
                CK(cudaMemcpy(a.p,input.data(),a.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(b.p,multiplier.data(),b.bytes,cudaMemcpyHostToDevice));
                root_launch(method,false,grid,t,a.get(),nullptr,tf.get(),0,stride);
                CK(cudaMemcpy(got.data(),a.p,a.bytes,cudaMemcpyDeviceToHost));compare(forward[method],got,fw);
                CK(cudaMemcpy(a.p,input.data(),a.bytes,cudaMemcpyHostToDevice));root_launch(method,true,grid,t,a.get(),b.get(),ti.get(),plan.scale,stride);
                CK(cudaMemcpy(got.data(),a.p,a.bytes,cudaMemcpyDeviceToHost));compare(inverse[method],got,iv);
                CK(cudaMemcpy(bgot.data(),b.p,b.bytes,cudaMemcpyDeviceToHost));compare(readonly[method],bgot,multiplier);
                CK(cudaMemcpy(a.p,input.data(),a.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(b.p,ones.data(),b.bytes,cudaMemcpyHostToDevice));
                root_launch(method,false,grid,t,a.get(),nullptr,tf.get(),0,stride);root_launch(method,true,grid,t,a.get(),b.get(),ti.get(),plan.scale,stride);
                CK(cudaMemcpy(got.data(),a.p,a.bytes,cudaMemcpyDeviceToHost));compare(roundtrip[method],got,input);
                std::vector<Word> gg,gw;
                for(unsigned int batch=0;batch<batches;++batch)for(unsigned int i=tile*tiles;i<stride;++i) {
                    gg.push_back(got[(size_t)batch*stride+i]);gw.push_back(input[(size_t)batch*stride+i]);
                }
                compare(guard[method],gg,gw);
            }
        }
    }
    Word bad=0;
    for(int method:{0,1,2,3})for(const auto group:{"forward","inverse","roundtrip","readonly","guard"}) {
        const auto &s=!std::strcmp(group,"forward") ? forward[method] : !std::strcmp(group,"inverse") ? inverse[method] :
            !std::strcmp(group,"roundtrip") ? roundtrip[method] : !std::strcmp(group,"readonly") ? readonly[method] : guard[method];
        bad+=s.bad;std::printf("small_root_check: method=%d group=%s cases=%llu words=%llu bad=%llu\n",method,group,s.cases,s.words,s.bad);
    }
    std::printf("small_root_tile_gate: bad=%llu\n",bad);return bad ? 3 : 0;
}
static int root_bench(int k,int t,int operation,int candidate)
{
    TileBuffers state(k,t,operation);root_attributes((1u<<t)*8);cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));int run=0;
    for(int mode:{0,1,1,0,1,0,0,1}) {
        fuse_fixture_env("NTT_GL_SHORT_REDUCE",mode && candidate==4 ? "1" : "0");ntt_gl_reduce_configure();
        const int method=mode && candidate!=4 ? candidate : 0;
        double seconds=0;Word wrong=0;
        for(int repeat=0;repeat<4;++repeat) {
            state.fill();CK(cudaEventRecord(start));
            root_launch(method,operation==1,dim3((unsigned int)(state.n>>t)),t,state.a.get(),state.b.get(),operation==1 ? state.ti.get() : state.tf.get(),state.plan.scale,state.n);
            if(operation==2)root_launch(method,true,dim3((unsigned int)(state.n>>t)),t,state.a.get(),state.b.get(),state.ti.get(),state.plan.scale,state.n);
            CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,start,end));if(repeat)seconds+=ms/1000.0;
            wrong=state.check();if(wrong)return 3;
        }
        std::printf("small_root_bench: run=%d mode=%d candidate=%d k=%d t=%d operation=%d N=%llu threads=512 seconds=%.9f bad=%llu\n",++run,mode,candidate,k,t,operation,state.n,seconds/3,wrong);
    }
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));return 0;
}
template<class Kernel>static void root_resources(int method,int inverse,Kernel kernel)
{
    cudaFuncAttributes a{};int blocks=0;CK(cudaFuncGetAttributes(&a,kernel));CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,512,32768));
    std::printf("small_root_resources: method=%d inverse=%d regs=%d local=%zu shared=32768 max_blocks=%d\n",method,inverse,a.numRegs,a.localSizeBytes,blocks);
}
int main(int argc,char **argv)
{
    const int device=argc>1 ? std::atoi(argv[1]) : 1;const char *mode=argc>2 ? argv[2] : "--check";CK(cudaSetDevice(device));
    std::setvbuf(stdout,nullptr,_IONBF,0);ntt_gl_reduce_configure();root_attributes(32768);
    root_resources(0,0,tile_kernel<false,true>);root_resources(0,1,tile_kernel<true,true>);
    root_resources(1,0,small_root_tile<1,false>);root_resources(1,1,small_root_tile<1,true>);
    root_resources(2,0,small_root_tile<2,false>);root_resources(2,1,small_root_tile<2,true>);
    root_resources(3,0,small_root_tile<3,false>);root_resources(3,1,small_root_tile<3,true>);
    int code=2;
    if(!std::strcmp(mode,"--check")){code=primitive_check(false);if(!code)code=reduce_check();if(!code)code=tile_check();}
    else if(!std::strcmp(mode,"--fault"))code=primitive_check(true);
    else if(!std::strcmp(mode,"--bench")) {
        const int k=argc>3 ? std::atoi(argv[3]) : 24,t=argc>4 ? std::atoi(argv[4]) : 12;
        const int operation=argc>5 ? std::atoi(argv[5]) : 0,candidate=argc>6 ? std::atoi(argv[6]) : 2;
        if(k>=12 && k<=26 && t>=5 && t<=12 && k>=t && operation>=0 && operation<=2 && candidate>=1 && candidate<=4)code=root_bench(k,t,operation,candidate);
    }
    CK(cudaDeviceSynchronize());std::printf("small_root_memory: requested_peak_bytes=%llu live_bytes=%llu\n",peak_bytes,live_bytes);return code ? code : live_bytes ? 4 : 0;
}
