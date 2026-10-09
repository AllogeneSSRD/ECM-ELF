#include "../../src/core/ecm_stage2_workspace_memory.h"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace ecm_stage2;
static unsigned checks=0;
static void check(bool v){++checks;if(!v)throw std::runtime_error("workspace assertion "+std::to_string(checks));}
static bool shape(Word m,int bits,S4ShapeDescriptor &d) {
    d.slots=2*m-1;d.bpw=8;d.slot_bits=2*bits+8;
    d.slot_words=(d.slot_bits+7)/8;d.n=1;
    while(d.n<d.slots*d.slot_words)d.n*=2;
    return true;
}
static bool layout(Word n,NttFuseMemoryLayout &d) {
    d.base={{NttBaseTileF,0,64},{NttBaseTileI,0,64},{NttBaseScratch,0,n/2}};
    d.table={{NttTablePassF,0,n/4},{NttTablePassI,0,n/4}};return true;
}
int main() {
    try {
        Word cases=0,compressed_cases=0,noncoincident=0;
        for(int bits:{31,127,521})for(Word p:{2ull,3ull,5ull,8ull,13ull,32ull})
        for(unsigned mask=0;mask<32;++mask)for(Word tail:{0ull,1ull}) {
            RequestProgram program;check(request_program(p,9*p+tail,program));
            NttMemoryPolicy np;np.pool=(mask&1)!=0;np.reuse_bq=(mask&2)!=0;np.carry_check=(mask&4)!=0;
            S4ProgramPolicy sp;sp.batch_bytes=1ull<<20;sp.chunk_max=2;
            sp.buffers=np.pool&&np.reuse_bq?2:3;sp.physical_chunks=true;
            sp.trim_raw=(mask&8)!=0;sp.trim_output=(mask&16)!=0;
            WorkspaceMemoryPlan a,b;Word peak=0,total=0,ntt=0,s4=0;
            check(workspace_memory_plan(program,bits,shape,layout,np,sp,a,false,
                [&](const WorkspaceMemoryEvent &e) {
                    Word &owner=e.ntt?ntt:s4;
                    if(e.allocation)owner+=e.bytes;else {check(owner>=e.bytes);owner-=e.bytes;}
                    check(owner==(e.ntt?e.ntt_live:e.s4_live));
                    total=ntt+s4;check(total==e.total);peak=std::max(peak,total);
                }));
            check(workspace_memory_plan(program,bits,shape,layout,np,sp,b));
            check(a.valid && a.finished && b.valid && b.finished && !total);
            check(a.peak_bytes==peak && a.peak_bytes==b.peak_bytes && a.final_bytes==b.final_bytes);
            check(a.ntt_at_peak+a.s4_at_peak==a.peak_bytes && !a.released_bytes && !b.released_bytes);
            NttMemoryPlan nm;
            check(ntt_memory_plan(program,bits,[](Word m,int bits,Word *n,Word *slots) {
                S4ShapeDescriptor d;shape(m,bits,d);*n=d.n;*slots=d.slots;return true;
            },layout,sp.batch_bytes,sp.physical_chunks,sp.chunk_max,np,nm));
            S4ProgramPlan sm;check(s4_program_plan(program,bits,shape,sp,sm));
            check(nm.valid && nm.finished && sm.valid);
            check(a.ntt_final.total==nm.final_payload.total && a.s4_final.total==sm.final_payload.total);
            check(a.ntt_peak_bytes==nm.peak_bytes && a.s4_peak_bytes==sm.peak_bytes);
            check(a.peak_bytes<=nm.peak_bytes+sm.peak_bytes);
            if(a.peak_bytes<nm.peak_bytes+sm.peak_bytes)++noncoincident;
            if(b.skipped_blocks)++compressed_cases;
            ++cases;
        }
        RequestProgram program;check(request_program(8,80,program));
        S4ProgramPolicy sp;sp.batch_bytes=1ull<<20;
        NttMemoryPolicy np;np.cap_bytes=128;
        WorkspaceMemoryPlan refusal;
        check(workspace_memory_plan(program,127,shape,layout,np,sp,refusal));
        check(refusal.valid && !refusal.finished && refusal.final_bytes>0);
        np.cap_bytes=0;sp.buffers=2;
        check(!workspace_memory_plan(program,127,shape,layout,np,sp,refusal));
        check(!refusal.valid);
        check(compressed_cases>0 && noncoincident>0);
        std::cout<<"{\"cases\":"<<cases<<",\"checks\":"<<checks<<",\"compressed_cases\":"<<compressed_cases
            <<",\"noncoincident_peaks\":"<<noncoincident<<",\"gpu_calls\":0}\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
