#pragma once
#include "ecm_stage2_geometry.h"
#include <functional>
#include <set>
#include <tuple>

// Successful S4 allocation events only: retained raw/output/packing buffers,
// reducer constants/selftests and the resident G-tree's metadata lease.
// No NTT, giant coordinates, fold/frontier owner, physical free-memory admission,
// CUDA context/events, pinned host memory or optional diagnostic fixtures.
namespace ecm_stage2 {
struct S4MemoryPayload {
    Word raw_a=0,raw_b=0,output=0,pack_a=0,pack_b=0;
    Word modulus=0,shape_constants=0,canonical_counter=0;
    Word selftest_digits=0,selftest_output=0,tree_metadata=0,total=0;
};
struct S4MemoryEvent {
    const char *owner=nullptr;
    bool allocation=false;
    Word bytes=0;
    S4MemoryPayload payload;
};
class S4MemoryState {
    Word words=0;
    bool tree_active=false;
    std::set<std::tuple<Word,Word,unsigned>> shapes;
    bool fail(const char *why){reason=why;return false;}
    bool refresh(const char *owner,bool allocation,Word bytes) {
        Word total=0;
        for(Word n:{live.raw_a,live.raw_b,live.output,live.pack_a,live.pack_b,
            live.modulus,live.shape_constants,live.canonical_counter,
            live.selftest_digits,live.selftest_output,live.tree_metadata})
            if(!add(total,n,total))return fail("payload_overflow");
        live.total=total;peak=std::max(peak,total);
        if(observe)observe({owner,allocation,bytes,live});
        return true;
    }
    bool release(Word &field,const char *owner) {
        const Word bytes=field;field=0;
        return !bytes || refresh(owner,false,bytes);
    }
    bool allocate(Word &field,Word bytes,const char *owner) {
        Word sum=0;if(field || !bytes)return fail("invalid_allocation");
        if(!add(live.total,bytes,sum))return fail("payload_overflow");
        field=bytes;return refresh(owner,true,bytes);
    }
    bool grow(Word &field,Word bytes,const char *owner) {
        return bytes<=field || (release(field,owner) && allocate(field,bytes,owner));
    }
public:
    S4MemoryPayload live;
    Word peak=0;
    const char *reason="ok";
    // One event per allocation/free, in native order. A future joint simulator
    // can interleave these with NTT/owner events; peak here is S4-only.
    std::function<void(const S4MemoryEvent &)> observe;
    bool init(Word w) {
        if(words || !w || w>max_words)return fail("invalid_word_count");
        words=w;return allocate(live.modulus,8*w,"reducer.modulus");
    }
    bool raw_reserve(Word a,Word b) {
        Word ba=0,bb=0;
        if(!multiply(a,8,ba) || !multiply(b,8,bb))return fail("payload_overflow");
        return grow(live.raw_a,ba,"raw.A") && grow(live.raw_b,bb,"raw.B");
    }
    bool raw_release() {return release(live.raw_a,"raw.A") && release(live.raw_b,"raw.B");}
    bool output_reserve(Word word_count) {
        Word bytes=0;if(!multiply(std::max(1ull,word_count),8,bytes))return fail("payload_overflow");
        return grow(live.output,bytes,"output");
    }
    bool output_release(){return release(live.output,"output");}
    bool pack_reserve(Word word_count) {
        Word bytes=0;if(!multiply(word_count,8,bytes))return fail("payload_overflow");
        if(bytes<=live.pack_a)return true;
        // The legacy temporary packer releases BOTH old arrays first.
        return release(live.pack_a,"pack.A") && release(live.pack_b,"pack.B") &&
            allocate(live.pack_a,bytes,"pack.A") && allocate(live.pack_b,bytes,"pack.B");
    }
    bool shape(Word slot_bits,Word slot_words,unsigned bpw) {
        Word stride=0;
        if(!words || !slot_bits || !slot_words || !bpw || bpw>=64 ||
           !multiply(slot_words,bpw,stride) || slot_bits>stride ||
           slot_bits<=stride-bpw)return fail("invalid_shape");
        const auto key=std::make_tuple(slot_bits,slot_words,bpw);
        if(shapes.count(key))return true;
        const Word constant=8*words;
        Word total=0;if(!add(live.shape_constants,constant,total) ||
            !add(live.total,constant,total))return fail("payload_overflow");
        live.shape_constants+=constant;
        if(!refresh("reducer.shape",true,constant))return false;
        shapes.insert(key);
        // Native mandatory selftest is 96 windows, BEFORE output/NTT growth;
        // its canonical counter argument is null, so it allocates no dbad.
        Word digits=0,output=0;
        if(!multiply(slot_words,96*8,digits) || !multiply(words,96*8,output))return fail("payload_overflow");
        return allocate(live.selftest_digits,digits,"selftest.digits") &&
            allocate(live.selftest_output,output,"selftest.output") &&
            release(live.selftest_digits,"selftest.digits") &&
            release(live.selftest_output,"selftest.output");
    }
    bool canonical_counter() {
        return live.canonical_counter || allocate(live.canonical_counter,8,"reducer.canonical");
    }
    Word shape_count()const{return (Word)shapes.size();}
    bool same_allocations(const S4MemoryState &other)const {
        return words==other.words && tree_active==other.tree_active && shapes==other.shapes &&
            live.raw_a==other.live.raw_a && live.raw_b==other.live.raw_b &&
            live.output==other.live.output && live.pack_a==other.live.pack_a && live.pack_b==other.live.pack_b &&
            live.modulus==other.live.modulus && live.shape_constants==other.live.shape_constants &&
            live.canonical_counter==other.live.canonical_counter &&
            live.selftest_digits==other.live.selftest_digits && live.selftest_output==other.live.selftest_output &&
            live.tree_metadata==other.live.tree_metadata;
    }
    bool tree_begin(Word n,bool compact_raw) {
        if(!words || !n || tree_active)return fail("invalid_tree_lease");
        Word a=0,b=0,pad=1,parents=n>1?n/2+(n%2!=0):0;
        if(!multiply(n,2,a) || !multiply(a,words,a) ||
           (!compact_raw && !multiply(n,2,b)) ||
           (compact_raw && parents && !add(n,parents,b)) ||
           !multiply(b,words,b) || !raw_reserve(a,b))return fail("payload_overflow");
        while(pad<n)if(!multiply(pad,2,pad))return fail("payload_overflow");
        Word bytes=0;
        if(pad>1 && !multiply(pad/2,24,bytes))return fail("payload_overflow");
        if(bytes && !allocate(live.tree_metadata,bytes,"tree.metadata"))return false;
        tree_active=true;return true;
    }
    bool tree_end(){tree_active=false;return release(live.tree_metadata,"tree.metadata");}
    bool close() {
        if(!tree_end())return false;
        for(size_t i=0;i<shapes.size();++i) {
            const Word bytes=8*words;live.shape_constants-=bytes;
            if(!refresh("reducer.shape",false,bytes))return false;
        }
        shapes.clear();
        // run_real declares S4Ctx before S4Reduce: reducer is destroyed first,
        // then S4Ctx frees output, raw A/B and temporary packed A/B.
        return release(live.modulus,"reducer.modulus") && release(live.canonical_counter,"reducer.canonical") &&
            output_release() && raw_release() && release(live.pack_a,"pack.A") && release(live.pack_b,"pack.B");
    }
};
} // namespace ecm_stage2
