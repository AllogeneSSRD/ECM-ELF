#include "runtime.h"
void NttGraphCache::destroy(Entry &entry) {
        if(entry.forward)CK(cudaGraphExecDestroy(entry.forward));
        if(entry.inverse)CK(cudaGraphExecDestroy(entry.inverse));
        entry.forward=entry.inverse=nullptr;
    }

void NttGraphCache::clear() {
        if(!entries.empty())++invalidations;
        for(auto &entry:entries)destroy(entry);
        entries.clear();
    }

void NttGraphCache::release() {
        clear();if(capture_stream){CK(cudaStreamDestroy(capture_stream));capture_stream=nullptr;}
    }

NttGraphCache::~NttGraphCache() {release();}

bool NttGraphCache::recoverable(cudaError_t error) {
        return error==cudaErrorMemoryAllocation || error==cudaErrorNotSupported ||
            error==cudaErrorStreamCaptureUnsupported || error==cudaErrorStreamCaptureInvalidated;
    }

bool NttGraphCache::accept(cudaError_t error) {
        if(error==cudaSuccess)return true;
        if(!recoverable(error))CK(error);
        ++failures;failed=true;(void)cudaGetLastError();
        stage2_log::print(stage2_log::debug,"ntt_graph_fallback: error=%s\n",cudaGetErrorString(error));return false;
    }

bool NttGraphCache::capture(cudaGraphExec_t &exec,bool inverse,const FuseCtx &fc,
        unsigned long long *a,unsigned long long *b,unsigned long long omega,
        unsigned long long omega_inv,unsigned long long scale,unsigned long long batch) {
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

NttGraphCache::Entry *NttGraphCache::lookup(int device,const FuseCtx &fc,unsigned long long *a,unsigned long long *b,
        unsigned long long omega,unsigned long long omega_inv,unsigned long long scale,unsigned long long batch) {
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

void NttGraphCache::print()const {
        stage2_log::print(stage2_log::phases,
            "ntt_graph_stats: requested=%d max_k=%llu max_words=%llu min_calls=%llu capacity=%llu candidates=%llu hits=%llu builds=%llu launches=%llu saved_launches=%llu skipped=%llu cold=%llu failures=%llu invalidations=%llu evictions=%llu entries=%llu peak_entries=%llu captured_nodes=%llu setup_seconds=%.6f\n",
            (int)enabled,max_k,max_words,min_calls,(unsigned long long)limit,candidates,hits,builds,launches,
            saved_launches,skipped,cold,failures,invalidations,evictions,(unsigned long long)entries.size(),peak_entries,nodes,setup_seconds);
    }
