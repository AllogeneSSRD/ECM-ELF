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
        Word cases=0,compressed_cases=0,noncoincident=0,giant_cases=0,giant_compressed=0,initial_cases=0;
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
            OwnerMemoryPolicy op{p,(Word)((bits+63)/64),1ull<<30,mask%4};
            WorkspaceMemoryPlan oa,ob;Word owner=0;ntt=s4=peak=total=0;
            check(workspace_memory_plan(program,bits,shape,layout,np,sp,oa,false,
                [&](const WorkspaceMemoryEvent &e) {
                    Word &v=e.owner_event?owner:e.ntt?ntt:s4;
                    if(e.allocation)v+=e.bytes;else {check(v>=e.bytes);v-=e.bytes;}
                    check(ntt==e.ntt_live && s4==e.s4_live && owner==e.owner_live);
                    total=ntt+s4+owner;check(total==e.total);peak=std::max(peak,total);
                },&op));
            check(workspace_memory_plan(program,bits,shape,layout,np,sp,ob,true,{},&op));
            check(oa.valid && oa.finished && ob.valid && ob.finished && !total && !owner);
            check(oa.peak_bytes==peak && oa.peak_bytes==ob.peak_bytes && oa.final_bytes==ob.final_bytes);
            check(oa.ntt_at_peak+oa.s4_at_peak+oa.owner_at_peak==oa.peak_bytes);
            check(oa.fold_bytes==owner_bytes(p,op.words,op.reuse) && oa.frontier_bytes==24*p);
            check(oa.owner_peak_bytes==oa.fold_bytes+oa.frontier_bytes);
            check(oa.peak_bytes<=a.peak_bytes+oa.owner_peak_bytes && oa.peak_bytes>=a.peak_bytes);
            for(Word trees:{1ull,3ull,4ull}) {
                const Word points=20*p+tail;
                RequestProgram joint;check(request_program(p,points,joint));
                GiantTimelinePolicy gp{p,points,op.words,trees*p,{}};
                gp.component.chain_min=2*p;gp.component.chain_block=4;
                gp.component.initial_points=3;gp.component.segment=2;gp.component.group=4;
                gp.component.seed_device=(mask&1)!=0;gp.component.seed_pair=(mask&2)!=0;
                gp.component.exact_segments=(mask&4)!=0;gp.component.resident_requested=(mask&8)!=0;
                gp.component.compact_products=(mask&16)!=0;
                WorkspaceMemoryPlan ga,gb;Word giant=0;owner=ntt=s4=total=peak=0;
                check(workspace_memory_plan(joint,bits,shape,layout,np,sp,ga,false,
                    [&](const WorkspaceMemoryEvent &e) {
                        Word &v=e.giant_event?giant:e.owner_event?owner:e.ntt?ntt:s4;
                        if(e.allocation)v+=e.bytes;else {check(v>=e.bytes);v-=e.bytes;}
                        check(ntt==e.ntt_live && s4==e.s4_live && owner==e.owner_live && giant==e.giant_live);
                        total=ntt+s4+owner+giant;check(total==e.total);peak=std::max(peak,total);
                    },&op,&gp));
                check(workspace_memory_plan(joint,bits,shape,layout,np,sp,gb,true,{},&op,&gp));
                check(ga.valid && ga.finished && gb.valid && gb.finished && !total);
                check(ga.peak_bytes==peak && ga.peak_bytes==gb.peak_bytes && ga.final_bytes==gb.final_bytes);
                check(ga.ntt_at_peak+ga.s4_at_peak+ga.owner_at_peak+ga.giant_at_peak==ga.peak_bytes);
                check(ga.points_consumed==points && gb.points_consumed==points);
                check(ga.point_chunks==ceil_ratio(points,trees*p) && gb.point_chunks==ga.point_chunks);
                check(!ga.released_bytes && !gb.released_bytes);
                GiantMemoryPlan gm;check(giant_memory_plan(p,points,op.words,trees*p,gp.component,gm));
                check(ga.giant_peak_bytes==gm.peak_bytes && gb.giant_peak_bytes==gm.peak_bytes);
                check(ga.giant_final_bytes==gm.accumulation_bytes && gb.giant_final_bytes==gm.accumulation_bytes);
                InitialMemoryPolicy ip{p,op.words,1ull<<30};
                WorkspaceMemoryPlan ia,ib;Word initial=0,fold_need=0,frontier_need=0;
                owner=ntt=s4=giant=total=peak=0;
                S4ShapeDescriptor target;check(shape(p+1,bits,target));
                const Word future_growth=np.pool?0:8*sp.buffers*target.n;
                check(workspace_memory_plan(joint,bits,shape,layout,np,sp,ia,false,
                    [&](const WorkspaceMemoryEvent &e) {
                        const bool fold_begin=e.owner_event && e.allocation && !owner;
                        Word &v=e.initial_event?initial:e.giant_event?giant:e.owner_event?owner:e.ntt?ntt:s4;
                        if(e.allocation)v+=e.bytes;else {check(v>=e.bytes);v-=e.bytes;}
                        check(ntt==e.ntt_live && s4==e.s4_live && owner==e.owner_live &&
                              giant==e.giant_live && initial==e.initial_live);
                        total=ntt+s4+owner+giant+initial;check(total==e.total);peak=std::max(peak,total);
                        if(fold_begin)fold_need=total-e.bytes+owner_bytes(p,op.words,op.reuse)+future_growth+(1ull<<30);
                        if(e.owner_event && e.allocation && owner==owner_bytes(p,op.words,op.reuse)+24*p)
                            frontier_need=total+future_growth+(1ull<<30);
                    },&op,&gp,&ip));
                check(workspace_memory_plan(joint,bits,shape,layout,np,sp,ib,true,{},&op,&gp,&ip));
                check(ia.valid && ia.finished && ib.valid && ib.finished && !total);
                check(ia.peak_bytes==peak && ia.peak_bytes==ib.peak_bytes && ia.final_bytes==ib.final_bytes);
                BabyMemoryLayout baby;check(baby_memory_layout(p,op.words,baby));
                check(ia.montgomery_bytes==8*op.words*(3*2048+1) && ia.baby_bytes==baby.bytes);
                check(ia.initial_peak_bytes==std::max(ia.montgomery_bytes,ia.baby_bytes));
                check(ia.peak_bytes==std::max(ga.peak_bytes,std::max(ia.montgomery_bytes,baby.bytes+8*op.words)));
                check(ia.ntt_at_peak+ia.s4_at_peak+ia.owner_at_peak+ia.giant_at_peak+ia.initial_at_peak==ia.peak_bytes);
                check(ia.baby_headroom_bytes==baby.bytes+8*op.words+(64ull<<20));
                check(ia.fold_headroom_bytes==fold_need && ia.frontier_headroom_bytes==frontier_need);
                ++initial_cases;
                if(gb.skipped_blocks)++giant_compressed;
                ++giant_cases;
            }
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
        sp.buffers=3;
        OwnerMemoryPolicy op{8,2,1,3};
        check(workspace_memory_plan(program,127,shape,layout,np,sp,refusal,true,{},&op));
        check(refusal.valid && !refusal.finished && !std::strcmp(refusal.reason,"fold_budget_refusal"));
        op.budget_bytes=owner_bytes(8,2,3);
        check(workspace_memory_plan(program,127,shape,layout,np,sp,refusal,true,{},&op));
        check(refusal.valid && !refusal.finished && !std::strcmp(refusal.reason,"frontier_budget_refusal"));
        check(compressed_cases>0 && noncoincident>0);
        check(giant_compressed>0);
        // Enormous B2 must collapse complete point-chunk cycles, preserving a
        // ladder tail whose larger seed capacity changes the final allocation.
        check(request_program(13,1300000000001ull,program));
        GiantTimelinePolicy gp{13,1300000000001ull,2,39,{}};
        gp.component.chain_min=30;gp.component.chain_block=4;
        sp.buffers=3;np.cap_bytes=0;op={13,2,1ull<<30,3};
        WorkspaceMemoryPlan huge;
        check(workspace_memory_plan(program,127,shape,layout,np,sp,huge,true,{},&op,&gp));
        check(huge.valid && huge.finished && huge.points_consumed==gp.points && huge.executed_blocks<40);
        check(huge.point_chunks==ceil_ratio(gp.points,gp.chunk_points) && huge.skipped_blocks>1000000000ull);
        InitialMemoryPolicy ip{13,2,0};
        check(workspace_memory_plan(program,127,shape,layout,np,sp,huge,true,{},&op,&gp,&ip));
        check(huge.valid && !huge.finished && !std::strcmp(huge.reason,"baby_budget_refusal") && !huge.executed_blocks);
        ip.baby_budget_bytes=1ull<<30;ip.diagnostic=true;
        check(workspace_memory_plan(program,127,shape,layout,np,sp,huge,true,{},&op,&gp,&ip));
        check(!huge.valid && !huge.finished && !std::strcmp(huge.reason,"diagnostic_initial_workspace_not_modeled"));
        std::cout<<"{\"cases\":"<<cases<<",\"checks\":"<<checks<<",\"compressed_cases\":"<<compressed_cases
            <<",\"noncoincident_peaks\":"<<noncoincident<<",\"giant_cases\":"<<giant_cases
            <<",\"giant_compressed\":"<<giant_compressed<<",\"initial_cases\":"<<initial_cases<<",\"gpu_calls\":0}\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
