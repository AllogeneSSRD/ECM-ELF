// Include the generated native allocator fixture, not the source template.
#define main s4_component_main
#include "fixture.cpp"
#undef main
#include "../../src/core/ecm_stage2_s4_program.h"
static Word program_cases=0,program_events=0;
static bool synthetic_shape(Word operand,int bits,S4ShapeDescriptor &d) {
    d.n=1;while(d.n<4*operand)d.n*=2;
    d.slots=2*operand-1;d.slot_bits=2*bits;
    for(Word n=operand;n>1;n=(n+1)/2)++d.slot_bits;
    d.bpw=25;d.slot_words=(d.slot_bits+24)/25;return true;
}
static void payload_check(const S4MemoryPayload &p,const S4Ctx &C,const Reducer &R) {
    check(p.raw_a==8*C.d_rawA_cap && p.raw_b==8*C.d_rawB_cap);
    check(p.output==8*C.d_out_cap && p.pack_a==0 && p.pack_b==0);
    check(p.modulus==8*R.w && p.shape_constants==8*R.w*R.shapes.size());
    check(p.canonical_counter==(R.dbad?8:0));check(p.tree_metadata==0);
    check(p.total==total);
}
static void program_case(Word p,Word points,int bits,S4ProgramPolicy policy) {
    check(heap.empty());actual.clear();total=peak=0;
    RequestProgram program;check(request_program(p,points,program));
    S4ProgramPlan model;check(s4_program_plan(program,bits,synthetic_shape,policy,model) && model.valid);
    Word requests=0,chunks=0,trees=0;
    {
        S4Ctx C;Reducer R;const Word w=(bits+63)/64;R.init(w);
        for(size_t bi=0;bi<program.blocks.size();++bi) {
            const auto &block=program.blocks[bi];
            if(bi==1 || bi+1==program.blocks.size()) {
                payload_check(bi==1?model.after_inverse:model.after_giant,C,R);
                if(policy.trim_raw)C.raw_release();if(policy.trim_output)C.output_release();
            }
            // Independently derive one G lease per middle block iteration,
            // including a tail of one leaf with no multiply requests.
            const Word leaves=(bi && bi+1<program.blocks.size())?
                ((bi+2==program.blocks.size() && points%p)?points%p:p):0;
            check(block.tree_leaves==leaves);
            for(Word iteration=0;iteration<block.repeat;++iteration) {
                Word pad=1;while(pad<leaves)pad*=2;
                bool in_tree=leaves!=0;void *meta=nullptr;
                if(in_tree) {
                    ++trees;std::vector<Word>deg((size_t)(2*pad));
                    for(Word i=0;i<leaves;++i)deg[(size_t)(pad+i)]=1;
                    for(Word i=pad;--i;)deg[(size_t)i]=deg[(size_t)(2*i)]+deg[(size_t)(2*i+1)];
                    Word b=0;for(Word i=pad/2;pad>1 && i<pad;++i)if(deg[(size_t)i])b+=(deg[(size_t)i]+1)*w;
                    C.raw_reserve((size_t)(2*leaves*w),(size_t)(policy.compact_raw?b:2*leaves*w));
                    if(pad>1)CK(cudaMalloc(&meta,(size_t)(3*(pad/2)*8)));
                }
                for(const auto &r:block.requests) {
                    if(in_tree && r.phase!=RequestGtrees){cudaFree(meta);meta=nullptr;in_tree=false;}
                    const unsigned source=(r.phase==RequestFtree || r.phase==RequestInverse)?RequestHost:
                        r.phase==RequestGtrees?RequestTreeRaw:
                        (r.phase==RequestFold || r.first==0)?RequestFoldOwner:RequestFrontierOwner;
                    check(r.input==source);
                    S4ShapeDescriptor d;const Word operand=std::max(r.ma,r.mb);
                    synthetic_shape(operand,bits,d);
                    Word c=1;
                    for(Word test=r.pairs;test;test/=2) {
                        const Word words=(policy.physical_chunks?policy.buffers:3)*d.n+d.slots+(policy.physical_chunks?2:0);
                        if(test*8*words<=policy.batch_bytes){c=test;break;}
                    }
                    if(policy.chunk_max)c=std::min(c,policy.chunk_max);
                    R.shape(d.slot_bits,d.slot_words,d.bpw);
                    output_reserve(C,(policy.chunk_output?c:r.pairs)*(policy.output_window?r.count:d.slots)*w);
                    for(Word first=0;first<r.pairs;first+=c) {
                        if(source==RequestHost)C.raw_reserve((size_t)(std::min(c,r.pairs-first)*operand*w),
                            (size_t)(std::min(c,r.pairs-first)*operand*w));
                        R.counter();++chunks;
                    }
                    ++requests;
                }
                if(meta)cudaFree(meta);
            }
        }
        payload_check(model.final_payload,C,R);check(model.shape_count==R.shapes.size());
    }
    check(total==0 && heap.empty());check(model.released_payload.total==0 && model.peak_bytes==peak);
    check(model.counters.requests==requests && model.counters.chunks==chunks && model.counters.trees==trees);
    Word allocations=0,frees=0;for(const auto &e:actual){if(e.alloc)++allocations;else++frees;}
    check(model.counters.allocations==allocations && model.counters.frees==frees);
    program_events+=actual.size();++program_cases;
}
int main() {
    try {
        for(int bits:{37,129,8193,16384})for(Word p:{1ull,3ull,63ull,64ull,65ull,129ull})
            for(Word points:{p+1,2*p+1,5*p+std::max(1ull,p/3)})for(unsigned mask=0;mask<8;++mask) {
                S4ProgramPolicy policy;policy.batch_bytes=16384;policy.buffers=mask&1?2:3;
                policy.physical_chunks=(mask&2)!=0;policy.compact_raw=(mask&4)!=0;
                policy.trim_raw=(mask&1)!=0;policy.trim_output=(mask&2)!=0;
                policy.chunk_max=mask&4?3:0;policy.output_window=(mask&1)!=0;policy.chunk_output=(mask&4)!=0;
                program_case(p,points,bits,policy);
            }
        RequestProgram huge;check(request_program(48,1ull<<55,huge));
        S4ProgramPolicy policy;policy.batch_bytes=16384;
        S4ProgramPlan large;check(s4_program_plan(huge,8193,synthetic_shape,policy,large) && large.valid);
        check(large.counters.trees==((1ull<<55)+47)/48 && large.skipped_blocks>1000000 && large.executed_blocks<16);
        RequestProgram bad;check(request_program(48,100,bad));bad.blocks[0].requests[0].input=RequestTreeRaw;
        S4ProgramPlan rejected;check(!s4_program_plan(bad,129,synthetic_shape,policy,rejected));
        std::cout<<"{\"checks\":"<<checks<<",\"program_cases\":"<<program_cases
            <<",\"allocation_events\":"<<program_events<<",\"huge_executed_blocks\":"<<large.executed_blocks
            <<",\"huge_skipped_blocks\":"<<large.skipped_blocks<<",\"bad\":0,\"gpu_calls\":0}\n";return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
