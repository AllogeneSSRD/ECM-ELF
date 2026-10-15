#pragma once

// Host-only cache. Nodes borrow arena arrays/tables; clear before any owner frees
// them. Capture uses a private stream, replay keeps the original stream order.
struct NttGraphCache {
    struct Key {
        unsigned long long v[64]={};
        bool operator==(const Key &other)const{return std::memcmp(v,other.v,sizeof(v))==0;}
    };
    struct Entry {
        Key key;unsigned long long seen=0,last=0;
        cudaGraphExec_t forward=nullptr,inverse=nullptr;
    };
    bool enabled=fuse_env_ull("NTT_SHORT_GRAPH",0)!=0,failed=false;
    unsigned long long max_k=fuse_env_ull("NTT_SHORT_GRAPH_MAX_K",19);
    unsigned long long max_words=fuse_env_ull("NTT_SHORT_GRAPH_MAX_WORDS",1ull<<22);
    unsigned long long min_calls=std::max(1ull,fuse_env_ull("NTT_SHORT_GRAPH_MIN_CALLS",4));
    size_t limit=(size_t)std::max(1ull,std::min(256ull,fuse_env_ull("NTT_SHORT_GRAPH_ENTRIES",64)));
    cudaStream_t capture_stream=nullptr;
    std::vector<Entry> entries;
    unsigned long long candidates=0,hits=0,builds=0,launches=0,skipped=0,cold=0;
    unsigned long long failures=0,invalidations=0,evictions=0,peak_entries=0,nodes=0,saved_launches=0;
    double setup_seconds=0;
    static void destroy(Entry &entry) {
        if(entry.forward)CK(cudaGraphExecDestroy(entry.forward));
        if(entry.inverse)CK(cudaGraphExecDestroy(entry.inverse));
        entry.forward=entry.inverse=nullptr;
    }
    void clear() {
        if(!entries.empty())++invalidations;
        for(auto &entry:entries)destroy(entry);
        entries.clear();
    }
    void release() {
        clear();if(capture_stream){CK(cudaStreamDestroy(capture_stream));capture_stream=nullptr;}
    }
    ~NttGraphCache(){release();}
    NttGraphCache()=default;
    NttGraphCache(const NttGraphCache &)=delete;
    NttGraphCache &operator=(const NttGraphCache &)=delete;
    static bool recoverable(cudaError_t error) {
        return error==cudaErrorMemoryAllocation || error==cudaErrorNotSupported ||
            error==cudaErrorStreamCaptureUnsupported || error==cudaErrorStreamCaptureInvalidated;
    }
    bool accept(cudaError_t error) {
        if(error==cudaSuccess)return true;
        if(!recoverable(error))CK(error);
        ++failures;failed=true;(void)cudaGetLastError();
        stage2_log::print(stage2_log::debug,"ntt_graph_fallback: error=%s\n",cudaGetErrorString(error));return false;
    }
    bool capture(cudaGraphExec_t &exec,bool inverse,const FuseCtx &fc,
        unsigned long long *a,unsigned long long *b,unsigned long long omega,
        unsigned long long omega_inv,unsigned long long scale,unsigned long long batch)
    {
        if(!accept(cudaStreamBeginCapture(capture_stream,cudaStreamCaptureModeThreadLocal)))return false;
        if(inverse)ntt_inverse_fused(a,b,fc,omega_inv,scale,batch,capture_stream);
        else {ntt_forward_fused(a,fc,omega,batch,capture_stream);ntt_forward_fused(b,fc,omega,batch,capture_stream);}
        cudaGraph_t graph=nullptr;const auto end=cudaStreamEndCapture(capture_stream,&graph);
        if(end!=cudaSuccess){if(graph)CK(cudaGraphDestroy(graph));return accept(end);}
        size_t count=0;CK(cudaGraphGetNodes(graph,nullptr,&count));
        const auto status=cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0);
        CK(cudaGraphDestroy(graph));if(!accept(status))return false;
        nodes+=count;return true;
    }
    Entry *lookup(int device,const FuseCtx &fc,unsigned long long *a,unsigned long long *b,
        unsigned long long omega,unsigned long long omega_inv,unsigned long long scale,unsigned long long batch)
    {
        if(!enabled || failed || fc.k<0 || (unsigned long long)fc.k>max_k || !batch ||
            fc.n>max_words/batch || !fc.tables_cached || !fc.arena_borrowed || fuse_trace_on()) {++skipped;return nullptr;}
        ++candidates;Key key;int i=0;
        key.v[i++]=(unsigned long long)device;key.v[i++]=fc.n;key.v[i++]=fc.k;key.v[i++]=fc.t;
        key.v[i++]=fc.m_max;key.v[i++]=fc.warp_tail;key.v[i++]=fc.coop_outer;key.v[i++]=fc.compact_scratch;
        key.v[i++]=omega;key.v[i++]=omega_inv;key.v[i++]=scale;key.v[i++]=batch;
        key.v[i++]=fuse_env_ull("NTT_GL_SHIFT_SCALE",0)!=0;
        key.v[i++]=(uintptr_t)a;key.v[i++]=(uintptr_t)b;key.v[i++]=(uintptr_t)fc.tblF;key.v[i++]=(uintptr_t)fc.tblI;
        for(int p=0;p<8;++p) {
            key.v[i++]=fc.ms[p];key.v[i++]=(uintptr_t)fc.passF[p];key.v[i++]=(uintptr_t)fc.radF[p];
            key.v[i++]=(uintptr_t)fc.passI[p];key.v[i++]=(uintptr_t)fc.radI[p];
        }
        Entry *found=nullptr;
        for(auto &entry:entries)if(entry.key==key){found=&entry;break;}
        if(!found) {
            if(entries.size()>=limit) {
                size_t victim=0;for(size_t p=1;p<entries.size();++p)if(entries[p].last<entries[victim].last)victim=p;
                destroy(entries[victim]);entries.erase(entries.begin()+(long)victim);++evictions;
            }
            entries.push_back(Entry{});found=&entries.back();found->key=key;
            peak_entries=std::max(peak_entries,(unsigned long long)entries.size());
        }
        auto &entry=*found;entry.last=candidates;++entry.seen;
        if(entry.forward && entry.inverse){++hits;return &entry;}
        if(entry.seen<min_calls){++cold;return nullptr;}
        size_t free=0,total=0;CK(cudaMemGetInfo(&free,&total));
        if(free<(64ull<<20)){++skipped;return nullptr;}
        const double begin=now_s();
        if(!capture_stream && !accept(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking)))return nullptr;
        if(!capture(entry.forward,false,fc,a,b,omega,omega_inv,scale,batch) ||
           !capture(entry.inverse,true,fc,a,b,omega,omega_inv,scale,batch)) {
            destroy(entry);setup_seconds+=now_s()-begin;return nullptr;
        }
        setup_seconds+=now_s()-begin;++builds;return &entry;
    }
    void print()const {
        stage2_log::print(stage2_log::phases,
            "ntt_graph_stats: requested=%d max_k=%llu max_words=%llu min_calls=%llu capacity=%llu candidates=%llu hits=%llu builds=%llu launches=%llu saved_launches=%llu skipped=%llu cold=%llu failures=%llu invalidations=%llu evictions=%llu entries=%llu peak_entries=%llu captured_nodes=%llu setup_seconds=%.6f\n",
            (int)enabled,max_k,max_words,min_calls,(unsigned long long)limit,candidates,hits,builds,launches,
            saved_launches,skipped,cold,failures,invalidations,evictions,(unsigned long long)entries.size(),peak_entries,nodes,setup_seconds);
    }
};
