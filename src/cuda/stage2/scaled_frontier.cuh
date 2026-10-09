// Production scaled descent. The final fold owner stays alive until the leaf
// readback; no arena/raw staging pointer is retained across a multiply.
struct ScaledFrontierDevice {
    FoldDeviceState *fold=nullptr;
    unsigned long long *metadata=nullptr;
    bool requested=false,enabled=false;
    const char *fallback="none";
    size_t metadata_words=0;
    unsigned long long metadata_bytes=0,owner_and_metadata_bytes=0;
    unsigned long long f_h2d_bytes=0,metadata_h2d_bytes=0,leaf_d2h_bytes=0,check_d2h_bytes=0;
    unsigned long long avoided_parent_h2d_bytes=0,avoided_state_d2h_bytes=0;
    unsigned long long logical_frontier_peak_bytes=0,host_staging_peak_bytes=0,pinned_borrow_peak_bytes=0;
    double setup_seconds=0,upload_seconds=0;
    ~ScaledFrontierDevice(){release();}
    void release(){if(metadata){CK(cudaFree(metadata));metadata=nullptr;}}
    bool init(FoldDeviceState &owner) {
        const double begin=now_s();fold=&owner;
        const size_t p=owner.P,w=owner.W;
        unsigned long long pw=0,need=0,bytes=0;
        if(!owner.active || !p || !w || owner.t!=0 ||
           !ecm_stage2::multiply(p,w,pw) || !ecm_stage2::multiply(pw,3,need) ||
           !ecm_stage2::multiply(p,3,bytes) || !ecm_stage2::multiply(bytes,8,bytes) ||
           !ecm_stage2::add(owner.stats->layout_bytes,bytes,owner_and_metadata_bytes) ||
           need>owner.memory.words[0] || need>owner.memory.words[1]) {fallback="owner_shape";return false;}
        metadata_bytes=bytes;metadata_words=3*p;
        auto max_mb=fuse_env_ull("NTT_FOLD_DEVICE_MAX_MB",640);
        max_mb=std::min(max_mb,fuse_env_ull("NTT_SCALED_FRONTIER_MAX_MB",max_mb));
        if(max_mb>(~0ull>>20) || owner_and_metadata_bytes>(max_mb<<20)) {fallback="budget";return false;}
        size_t available=0,total=0;CK(cudaMemGetInfo(&available,&total));
        unsigned long long qn=0;
        if(!ntt_shape_query(p+1,(int)owner.layer->S,&qn,nullptr,nullptr,nullptr,nullptr,nullptr)) {fallback="ntt_shape";return false;}
        const unsigned buffers=owner.layer->arena->big_buffer_count();
        const size_t target=8ull*buffers*qn,current=8ull*owner.layer->arena->workspace.words;
        const size_t growth=target>current?target-current:0;
        const size_t future_reserve=1ull<<30;
        const bool fits=bytes<=available && available-bytes>=growth+future_reserve;
        stage2_log::print(stage2_log::debug,
            "frontier_device_headroom: available_bytes=%llu metadata_bytes=%llu physical_buffers=%u target_big_bytes=%llu current_big_bytes=%llu growth_bytes=%llu future_reserve_bytes=%llu fits=%d\n",
            (unsigned long long)available,bytes,buffers,(unsigned long long)target,
            (unsigned long long)current,(unsigned long long)growth,(unsigned long long)future_reserve,(int)fits);
        if(!fits) {fallback="headroom";return false;}
        if(gscale_flag("NTT_SCALED_FRONTIER_ALLOC_FAIL")) {fallback="allocation_fixture";return false;}
        const auto error=cudaMalloc(&metadata,(size_t)bytes);
        if(error==cudaErrorMemoryAllocation){cudaGetLastError();metadata=nullptr;fallback="allocation";return false;}
        CK(error);enabled=true;setup_seconds=now_s()-begin;return true;
    }
    // Input F stays in ordinary order on the host. The reverse gather packs it
    // directly into NTT digits. Existing pinned output slots are borrowed only
    // after their recorded D2H/H2D consumer has completed.
    void upload_siblings(unsigned long long *dst,const std::vector<std::vector<unsigned long long>> &Ft,
        size_t nbase,const std::vector<std::pair<size_t,size_t>> &group,size_t mb,size_t w) {
        const double begin=now_s();const size_t slice=mb*w,words=group.size()*slice;
        bool available[2]={false,false},used[2]={false,false};
        if(g_s4_async)for(int i=0;i<2;++i)available[i]=g_pin_out[i] && g_pin_out_cap[i] && g_pin_ev[i];
        auto copy=[&](unsigned long long *out,size_t first,size_t count) {
            while(count) {
                const size_t s=first/slice,offset=first%slice,take=std::min(count,slice-offset);
                const auto &f=Ft[(nbase+group[s].second)^1];
                if(f.size()!=slice){std::fprintf(stderr,"FATAL: scaled frontier sibling length\n");std::exit(3);}
                std::copy_n(f.data()+offset,take,out);out+=take;first+=take;count-=take;
            }
        };
        if(available[0] || available[1]) {
            size_t offset=0,turn=0;
            while(offset<words) {
                size_t i=turn++&1;if(!available[i])i^=1;
                CK(cudaEventSynchronize(g_pin_ev[i]));
                const size_t take=std::min(words-offset,g_pin_out_cap[i]);
                copy(g_pin_out[i],offset,take);
                CK(cudaMemcpyAsync(dst+offset,g_pin_out[i],take*8,cudaMemcpyHostToDevice));
                CK(cudaEventRecord(g_pin_ev[i]));used[i]=true;offset+=take;
                pinned_borrow_peak_bytes=std::max(pinned_borrow_peak_bytes,8ull*take);
            }
            for(int i=0;i<2;++i)if(used[i])CK(cudaEventSynchronize(g_pin_ev[i]));
        } else {
            std::vector<unsigned long long> packed(words);copy(packed.data(),0,words);
            CK(cudaMemcpy(dst,packed.data(),words*8,cudaMemcpyHostToDevice));
            host_staging_peak_bytes=std::max(host_staging_peak_bytes,8ull*packed.capacity());
        }
        f_h2d_bytes+=8ull*words;upload_seconds+=now_s()-begin;
    }
    void run(PolyLayer &L,const std::vector<std::vector<unsigned long long>> &Ft,
        const std::vector<size_t> &deg,size_t pad,const CPoly &H,
        std::vector<std::vector<unsigned long long>> &values,ScaledStats &st,int cat,bool check) {
        const size_t p=fold->P,w=fold->W,pw=p*w;
        if(!enabled || !metadata || !fold->active || deg.size()<2*pad || Ft.size()<2*pad || deg[1]!=p ||
           (check && p>512)) {std::fprintf(stderr,"FATAL: scaled frontier live shape/check scope\n");std::exit(3);}
        auto *cur=fold->memory.data[1],*next=fold->memory.data[0];
        std::vector<unsigned long long> offsets(1,0);
        ++st.root_inverse_reused;++st.mul_calls;++st.mul_pairs;
        const bool poison=gscale_flag("NTT_SCALED_FRONTIER_TEST_BAD");
        if(poison && !check){std::fprintf(stderr,"FATAL: scaled frontier poison requires GMP check\n");std::exit(3);}
        auto corrupt=[&](unsigned long long *dst) {
            unsigned long long v=0;CK(cudaMemcpy(&v,dst,8,cudaMemcpyDeviceToHost));v^=1;
            CK(cudaMemcpy(dst,&v,8,cudaMemcpyHostToDevice));check_d2h_bytes+=8;
        };
        auto validate=[&](size_t base) {
            std::vector<unsigned long long> actual;
            if(check){actual.resize(pw);CK(cudaMemcpy(actual.data(),cur,pw*8,cudaMemcpyDeviceToHost));check_d2h_bytes+=pw*8;}
            size_t words=0;
            for(size_t i=0;i<base;++i) {
                const size_t d=deg[base+i];
                if(offsets[i]!=words || d>p-words/w){std::fprintf(stderr,"FATAL: scaled frontier degree/offset\n");std::exit(3);}
                ++st.states;st.words+=d*w;words+=d*w;
                if(check) {
                    const auto expected=scaled_state_gmp(H,Ft[base+i],d,L);
                    if(!std::equal(expected.begin(),expected.end(),actual.begin()+offsets[i])) {
                        std::fprintf(stderr,"FATAL: scaled frontier GMP node mismatch node=%llu degree=%llu\n",
                            (unsigned long long)(base+i),(unsigned long long)d);std::exit(3);
                    }
                    ++st.checked_states;st.checked_words+=d*w;
                }
            }
            if(words!=pw){std::fprintf(stderr,"FATAL: scaled frontier degree conservation\n");std::exit(3);}
        };
        if(poison && pad==1)corrupt(cur);
        validate(1);
        for(size_t base=1;base<pad;base*=2) {
            const size_t nbase=2*base;
            std::vector<unsigned long long> target(nbase);size_t words=0;
            for(size_t i=0;i<nbase;++i){target[i]=words;words+=deg[nbase+i]*w;}
            if(words!=pw){std::fprintf(stderr,"FATAL: scaled frontier next length\n");std::exit(3);}
            logical_frontier_peak_bytes=std::max(logical_frontier_peak_bytes,16ull*pw);
            std::map<std::pair<size_t,size_t>,std::vector<std::pair<size_t,size_t>>> groups;
            for(size_t j=0;j<base;++j)for(size_t side=0;side<2;++side) {
                const size_t slot=2*j+side,ci=nbase+slot,a=deg[ci],b=deg[ci^1];
                if(a+b!=deg[base+j]){std::fprintf(stderr,"FATAL: scaled frontier child degrees\n");std::exit(3);}
                if(!a){++st.zeros;continue;}
                if(!b){CK(cudaMemcpyAsync(next+target[slot],cur+offsets[j],a*w*8,cudaMemcpyDeviceToDevice));++st.copies;continue;}
                groups[{a,b}].push_back({j,slot});
            }
            for(const auto &g:groups) {
                const size_t a=g.first.first,b=g.first.second,ma=a+b,mb=b+1,nb=g.second.size();
                if(nb>p || mb>2*p/nb){std::fprintf(stderr,"FATAL: scaled frontier sibling budget\n");std::exit(3);}
                std::vector<unsigned long long> map(3*nb);
                for(size_t s=0;s<nb;++s){map[s]=offsets[g.second[s].first];map[nb+s]=pw+s*mb*w;map[2*nb+s]=target[g.second[s].second];}
                upload_siblings(cur+pw,Ft,nbase,g.second,mb,w);
                CK(cudaMemcpy(metadata,map.data(),map.size()*8,cudaMemcpyHostToDevice));metadata_h2d_bytes+=map.size()*8;
                S4DeviceBatch batch{cur,metadata,next,map.data(),pw+nb*mb*w,pw,nb,&fold->memory,true};
                std::vector<unsigned long long> unused;
                poly_mul_batch_modN(L,nullptr,nullptr,ma,mb,nb,unused,cat,nullptr,b,a,&batch);
                ++st.mul_calls;st.mul_pairs+=nb;
                avoided_parent_h2d_bytes+=8ull*nb*ma*w;avoided_state_d2h_bytes+=8ull*nb*a*w;
            }
            if(poison && base==1)corrupt(next);
            std::swap(cur,next);offsets.swap(target);++st.levels;validate(nbase);
        }
        std::vector<unsigned long long> leaves(pw);CK(cudaMemcpy(leaves.data(),cur,pw*8,cudaMemcpyDeviceToHost));leaf_d2h_bytes=pw*8;
        values.assign(p,std::vector<unsigned long long>(w));size_t leaf=0;
        for(size_t i=0;i<pad;++i)if(deg[pad+i]) {
            if(deg[pad+i]!=1 || leaf>=p){std::fprintf(stderr,"FATAL: scaled frontier nonlinear leaf\n");std::exit(3);}
            std::copy_n(leaves.data()+offsets[i],w,values[leaf++].data());
        }
        if(leaf!=p){std::fprintf(stderr,"FATAL: scaled frontier leaf count\n");std::exit(3);}
        st.leaves+=leaf;
        st.frontier_peak_bytes=logical_frontier_peak_bytes;
        st.pack_peak_bytes=host_staging_peak_bytes;
    }
    void print()const {
        stage2_log::print(stage2_log::phases, "scaled_frontier_device: requested=%d enabled=%d metadata_bytes=%llu owner_and_metadata_bytes=%llu f_h2d_bytes=%llu metadata_h2d_bytes=%llu leaf_d2h_bytes=%llu check_d2h_bytes=%llu avoided_parent_h2d_bytes=%llu avoided_state_d2h_bytes=%llu logical_frontier_peak_bytes=%llu host_staging_peak_bytes=%llu pinned_borrow_peak_bytes=%llu setup_seconds=%.6f upload_seconds=%.6f fallback=%s\n",
            (int)requested,(int)enabled,metadata_bytes,owner_and_metadata_bytes,f_h2d_bytes,metadata_h2d_bytes,leaf_d2h_bytes,check_d2h_bytes,
            avoided_parent_h2d_bytes,avoided_state_d2h_bytes,logical_frontier_peak_bytes,host_staging_peak_bytes,pinned_borrow_peak_bytes,setup_seconds,upload_seconds,fallback);
    }
};
