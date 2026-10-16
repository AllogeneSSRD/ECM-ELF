// Included inside cgbn_edwards_instances.cuh's private namespace, after put().
struct EdwardsEvent {
    cudaEvent_t value{};
    EdwardsEvent(){check(cudaEventCreate(&value));}
    ~EdwardsEvent(){cudaEventDestroy(value);}
};
template<class P> int run_naf(const mpz_t n,const EcmEdwardsNaf &plan,mpz_t *states,
        uint32_t count,uint32_t &done,uint32_t chunk,EcmEdwardsProgress cb,
        void *context,int doubling,EcmEdwardsNafMetrics *metrics) {
    using Mem=cgbn_mem_t<P::BITS>;
    const uint32_t entries=1u<<(plan.window-2),ipb=P::TPB/P::TPI;
    if(metrics){*metrics={};metrics->columns=plan.bits;metrics->additions=plan.additions;
        metrics->container_bits=P::BITS;metrics->tpi=P::TPI;
        metrics->dictionary_bytes=uint64_t(count)*entries*4*sizeof(Mem);}
    if(done==plan.bits)return 0;
    std::vector<Mem> host(size_t(count)*10),ordinary(size_t(count)*4);
    for(uint32_t i=0;i<count;++i) {
        put(host[size_t(i)*10],n);
        for(unsigned j=0;j<9;++j)put(host[size_t(i)*10+j+1],states[size_t(i)*9+j]);
    }
    Device<Mem> data(host.size()),result(ordinary.size()),dictionary(size_t(count)*entries*4);
    Device<uint32_t> nonzero(plan.nonzero.size()),codes(plan.codes.size());Report report;
    check(cudaMemcpy(data.p,host.data(),host.size()*sizeof(Mem),cudaMemcpyHostToDevice));
    check(cudaMemcpy(nonzero.p,plan.nonzero.data(),plan.nonzero.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(codes.p,plan.codes.data(),plan.codes.size()*4,cudaMemcpyHostToDevice));
    EdwardsEvent start,stop;
    check(cudaEventRecord(start.value));
    edwards_naf_prepare<P><<<(count+ipb-1)/ipb,P::TPB>>>(report.p,data.p,dictionary.p,entries,count);
    check(cudaGetLastError());check(cudaEventRecord(stop.value));check(cudaEventSynchronize(stop.value));
    if(cgbn_error_report_check(report.p))throw std::runtime_error("CGBN Edwards dictionary arithmetic error");
    if(metrics)check(cudaEventElapsedTime(&metrics->dictionary_gpu_ms,start.value,stop.value));
    while(done<plan.bits) {
        const uint32_t end=done+std::min(chunk,plan.bits-done);
        check(cudaEventRecord(start.value));
        edwards_naf_slice<P><<<(count+ipb-1)/ipb,P::TPB>>>(report.p,data.p,dictionary.p,result.p,
            nonzero.p,codes.p,plan.bits,done,end,entries,plan.window,count,doubling);
        check(cudaGetLastError());check(cudaEventRecord(stop.value));check(cudaEventSynchronize(stop.value));
        if(cgbn_error_report_check(report.p))throw std::runtime_error("CGBN Edwards NAF arithmetic error");
        if(metrics){float ms=0;check(cudaEventElapsedTime(&ms,start.value,stop.value));metrics->loop_gpu_ms+=ms;++metrics->launches;}
        check(cudaMemcpy(ordinary.data(),result.p,ordinary.size()*sizeof(Mem),cudaMemcpyDeviceToHost));
        for(uint32_t i=0;i<count;++i)for(unsigned j=0;j<4;++j)
            mpz_import(states[size_t(i)*9+j+5],P::BITS/32,-1,4,0,0,ordinary[size_t(i)*4+j]._limbs);
        done=end;if(cb && !cb(done,context))return 2;
    }
    return 0;
}
